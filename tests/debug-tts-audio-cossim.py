#!/usr/bin/env python3
# debug-tts-audio-cossim.py: compare the C++ TTS pipeline output WAV against
# the Python reference WAV produced inline by chaining _generate_iterative
# (class_temperature=0, position_temperature=0) and audio_tokenizer.decode.
# Both sides skip post-processing (no RMS adjust, no fade, no trim) so the
# two outputs are directly comparable in the time and mel domains. Bytewise
# tokens are not expected to match because the LLM forward already diverges
# at the argmax level; the audio cossim is the validation that matters at
# the perceptual level.
#
# The Python model loads once and runs the 3 cases inline, no subprocess.
# The C++ binary reads target text from stdin and the reference transcript
# from a temp file.
#
# Usage:
#   ./tests/debug-tts-audio-cossim.py

import os
import struct
import subprocess
import sys

import numpy as np

try:
    import librosa
except ImportError:
    print("error: librosa not installed", file=sys.stderr); sys.exit(1)

import torch

sys.path.insert(0, "../OmniVoice")
from omnivoice.models.omnivoice import (
    GenerationTask,
    OmniVoice,
    OmniVoiceGenerationConfig,
)

BIN            = "../build/omnivoice-tts"
MODEL_LM       = "../models/omnivoice-base-F32.gguf"
MODEL_CODEC    = "../models/omnivoice-tokenizer-F32.gguf"
CHECKPOINT_DIR = "../checkpoints/OmniVoice"

CASES = [
    {"text": "Hello world.", "lang": "English", "instruct": "Speak softly",
     "duration": 16, "denoise": False,
     "ref_wav": None, "ref_text": None},
    {"text": "Voici une phrase en francais.", "lang": "French",
     "instruct": "Calmly", "duration": 24, "denoise": False,
     "ref_wav": None, "ref_text": None},
    {"text": "This is a synthesized sentence.", "lang": "English",
     "instruct": "Calmly", "duration": 24, "denoise": True,
     "ref_wav": "codec-ref.wav",
     "ref_text": "Reference audio transcription placeholder."},
]

SAMPLE_RATE = 24000
MEL_HOP     = 240
MEL_N_FFT   = 1024
MEL_N_MELS  = 80
THRESHOLD   = 0.80

def write_wav_mono_f32(path, audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float32)
    n = audio.size
    byte_rate = sample_rate * 4
    block     = 4
    fmt_code  = 3
    data_size = n * 4
    riff_size = 36 + data_size
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, fmt_code, 1, sample_rate, byte_rate, block, 32))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio.tobytes())

def read_wav_mono_f32(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RuntimeError("not a RIFF WAVE: %s" % path)
    pos = 12
    fmt_code = n_ch = sr = bits = None
    samples = None
    while pos + 8 <= len(data):
        cid = data[pos:pos+4]
        csize = struct.unpack("<I", data[pos+4:pos+8])[0]
        body = data[pos+8:pos+8+csize]
        if cid == b"fmt ":
            fmt_code = struct.unpack("<H", body[0:2])[0]
            n_ch     = struct.unpack("<H", body[2:4])[0]
            sr       = struct.unpack("<I", body[4:8])[0]
            bits     = struct.unpack("<H", body[14:16])[0]
        elif cid == b"data":
            if fmt_code == 3 and bits == 32:
                samples = np.frombuffer(body, dtype="<f4").copy()
            else:
                raise RuntimeError("unsupported WAV: fmt_code=%s bits=%s" % (fmt_code, bits))
            break
        pos += 8 + csize + (csize & 1)
    if samples is None:
        raise RuntimeError("no data chunk in %s" % path)
    if n_ch != 1:
        raise RuntimeError("expected mono, got %d channels" % n_ch)
    return sr, samples

def waveform_cossim(a, b):
    n = min(a.size, b.size)
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    rms_a = np.sqrt(np.mean(a * a) + 1e-12)
    rms_b = np.sqrt(np.mean(b * b) + 1e-12)
    a = a / rms_a
    b = b / rms_b
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def log_mel(audio, sr):
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=sr,
        n_fft=MEL_N_FFT,
        hop_length=MEL_HOP,
        n_mels=MEL_N_MELS,
        power=2.0,
    )
    return np.log(mel + 1e-6)

def mel_cossim(a, b, sr):
    mel_a = log_mel(a, sr)
    mel_b = log_mel(b, sr)
    n = min(mel_a.shape[1], mel_b.shape[1])
    fa = mel_a[:, :n].flatten().astype(np.float64)
    fb = mel_b[:, :n].flatten().astype(np.float64)
    return float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-12))

def encode_ref_wav(model, ref_wav_path):
    # Encode the reference WAV / MP3 the same way the upstream voice-clone
    # path does : resample / downmix to mono 24 kHz, hop-trim, RVQ encode.
    tok_sr = model.audio_tokenizer.config.sample_rate
    wav, _ = librosa.load(ref_wav_path, sr=tok_sr, mono=True)
    wav = wav.astype(np.float32)
    hop = model.audio_tokenizer.config.hop_length
    clip = int(wav.shape[-1] % hop)
    if clip > 0:
        wav = wav[:-clip]
    wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(model.audio_tokenizer.device)
    with torch.no_grad():
        ref_audio_tokens = model.audio_tokenizer.encode(wav_t).audio_codes.squeeze(0)
    return ref_audio_tokens

def run_py(model, case, out_path):
    K = model.config.num_audio_codebook
    T = case["duration"]
    ref_audio_tokens = None
    if case.get("ref_wav"):
        ref_wav_path = case["ref_wav"]
        ref_audio_tokens = encode_ref_wav(model, ref_wav_path)
    task = GenerationTask(
        batch_size=1,
        texts=[case["text"]],
        target_lens=[T],
        langs=[case["lang"]],
        instructs=[case["instruct"]],
        ref_texts=[case["ref_text"]],
        ref_audio_tokens=[ref_audio_tokens],
        ref_rms=[None],
    )
    cfg = OmniVoiceGenerationConfig(
        num_step=32,
        guidance_scale=2.0,
        t_shift=0.1,
        layer_penalty_factor=5.0,
        position_temperature=0.0,
        class_temperature=0.0,
        denoise=case["denoise"],
    )
    with torch.no_grad():
        outs = model._generate_iterative(task, cfg)
    tokens = outs[0]
    if tokens.shape[0] != K or tokens.shape[1] != T:
        raise RuntimeError("unexpected token shape %s (expected (%d, %d))" %
                           (tuple(tokens.shape), K, T))
    tok_dev = model.audio_tokenizer.device
    with torch.no_grad():
        audio = (
            model.audio_tokenizer.decode(tokens.to(tok_dev).unsqueeze(0))
            .audio_values[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio[0]
    sample_rate = model.audio_tokenizer.config.sample_rate
    write_wav_mono_f32(out_path, audio, sample_rate)

def run_cpp(case, out_path):
    args = [BIN, "--model", MODEL_LM, "--codec", MODEL_CODEC,
            "--duration", str(case["duration"]),
            "-o", out_path]
    if case["lang"] is not None:
        args += ["--lang", case["lang"]]
    if case["instruct"] is not None:
        args += ["--instruct", case["instruct"]]
    if not case["denoise"]:
        args += ["--no-denoise"]
    ref_text_tmp = None
    if case.get("ref_wav"):
        ref_wav_path = case["ref_wav"]
        ref_text_tmp = "tts_ref_text.txt"
        with open(ref_text_tmp, "w") as f:
            f.write(case["ref_text"])
        args += ["--ref-wav", ref_wav_path, "--ref-text", ref_text_tmp]
    try:
        return subprocess.run(args, input=case["text"], text=True,
                              check=False, capture_output=False).returncode
    finally:
        if ref_text_tmp is not None and os.path.exists(ref_text_tmp):
            os.unlink(ref_text_tmp)

def main():
    if not os.path.exists(BIN):
        print("error: build omnivoice-tts first", file=sys.stderr); sys.exit(1)
    if not os.path.isdir(CHECKPOINT_DIR):
        print("error: %s not found" % CHECKPOINT_DIR, file=sys.stderr); sys.exit(1)

    print("loading OmniVoice from %s ..." % CHECKPOINT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OmniVoice.from_pretrained(
        CHECKPOINT_DIR,
        attn_implementation="sdpa",
        torch_dtype=torch.float32,
    ).to(device).eval()

    n_pass = 0
    n_fail = 0
    for i, case in enumerate(CASES):
        cpp_path = "tts_cpp.wav"
        py_path  = "tts_py.wav"
        try:
            print("[%d] case=%s" % (i, case))
            print("[%d] running cpp ..." % i)
            rc_cpp = run_cpp(case, cpp_path)
            if rc_cpp != 0:
                print("[%d] CRASH cpp=%d" % (i, rc_cpp))
                n_fail += 1
                continue
            print("[%d] running py ..." % i)
            run_py(model, case, py_path)
            sr_c, audio_c = read_wav_mono_f32(cpp_path)
            sr_p, audio_p = read_wav_mono_f32(py_path)
            print("[%d] cpp: %d samples @ %d Hz  range=[%.4f, %.4f]  rms=%.4f" %
                  (i, audio_c.size, sr_c, audio_c.min(), audio_c.max(),
                   float(np.sqrt(np.mean(audio_c * audio_c)))))
            print("[%d] py : %d samples @ %d Hz  range=[%.4f, %.4f]  rms=%.4f" %
                  (i, audio_p.size, sr_p, audio_p.min(), audio_p.max(),
                   float(np.sqrt(np.mean(audio_p * audio_p)))))
            if sr_c != sr_p:
                print("[%d] FAIL sample rate mismatch %d vs %d" % (i, sr_c, sr_p))
                n_fail += 1
                continue
            wave_cs = waveform_cossim(audio_c, audio_p)
            mel_cs  = mel_cossim(audio_c, audio_p, sr_c)
            print("[%d] waveform cossim (RMS-norm): %.4f" % (i, wave_cs))
            print("[%d] log-mel  cossim           : %.4f" % (i, mel_cs))
            if mel_cs >= THRESHOLD:
                print("[%d] OK   mel cossim >= %.2f" % (i, THRESHOLD))
                n_pass += 1
            else:
                print("[%d] FAIL mel cossim %.4f < threshold %.2f" % (i, mel_cs, THRESHOLD))
                n_fail += 1
        finally:
            for p in (cpp_path, py_path):
                if os.path.exists(p):
                    os.unlink(p)

    print("[Summary] %d pass / %d fail / %d total  (mel cossim threshold %.2f)" %
          (n_pass, n_fail, len(CASES), THRESHOLD))
    sys.exit(0 if n_fail == 0 else 1)

if __name__ == "__main__":
    main()
