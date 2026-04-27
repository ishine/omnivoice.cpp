#!/usr/bin/env python3
"""Validate the C++ pipeline_codec_encode end-to-end against PyTorch and via
a roundtrip through pipeline_codec_decode.

Three checks :
    1. codes_cpp vs codes_pytorch        - exact match rate per codebook
    2. roundtrip_cpp.wav vs codec-ref.wav - audio cossim (post BF16 + RVQ
                                            quantization, 0.85+ expected)
    3. roundtrip_cpp.wav vs roundtrip_pt.wav - audio cossim of the C++
                                               encode-then-decode chain vs
                                               the upstream encode-then-decode
                                               chain (both losing the same
                                               quantization, ideally near 1)

Usage :
    ./debug-encode-roundtrip-cossim.py
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "../OmniVoice")
from omnivoice.models.omnivoice import OmniVoice

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    n    = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d    = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0

def norm_ratio(a, b):
    na = np.linalg.norm(a.astype(np.float64))
    nb = np.linalg.norm(b.astype(np.float64))
    return float(na / nb) if nb > 1e-10 else 0.0

def stats(name, cpp, ref):
    diff = np.abs(cpp.flatten() - ref.flatten())
    print(f"{name:30s} cos={cos(cpp, ref):.6f} norm_ratio={norm_ratio(cpp, ref):.4f} maxdiff={diff.max():.4e} meandiff={diff.mean():.4e}")

def read_codes(path, num_codebooks):
    BITS = 11
    MASK = (1 << BITS) - 1
    with open(path, "rb") as f:
        data = f.read()
    total_bits = len(data) * 8
    n_codes    = total_bits // BITS
    n_frames   = n_codes // num_codebooks
    out = np.zeros(n_codes, dtype=np.int32)
    acc          = 0
    bits_in_acc  = 0
    in_pos       = 0
    for i in range(n_codes):
        while bits_in_acc < BITS and in_pos < len(data):
            acc |= data[in_pos] << bits_in_acc
            bits_in_acc += 8
            in_pos += 1
        out[i] = acc & MASK
        acc >>= BITS
        bits_in_acc -= BITS
    return out.reshape(num_codebooks, n_frames)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="codec-ref.wav")
    ap.add_argument("--bin", default="../build/omnivoice-codec")
    ap.add_argument("--model", default="../models/omnivoice-tokenizer-F32.gguf")
    ap.add_argument("--checkpoints", default="../checkpoints/OmniVoice")
    args = ap.parse_args()

    if not os.path.isfile(args.bin):
        print(f"[main] binary not found : {args.bin}", file=sys.stderr)
        sys.exit(1)

    wav, sr = sf.read(args.wav)
    if wav.ndim > 1:
        wav = wav[:, 0]
    print(f"[ref] wav : {len(wav)} samples sr={sr}")

    # PyTorch reference : tok.encode then tok.decode.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OmniVoice.from_pretrained(
        args.checkpoints,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device).eval()
    tok = model.audio_tokenizer

    audio_24k_t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)  # (B, C=1, N)
    with torch.inference_mode():
        codes_pt = tok.encode(audio_24k_t).audio_codes.cpu().numpy().astype(np.int32)  # (B, CB, T)
    codes_pt = codes_pt[0]  # (CB, T)
    print(f"[ref] codes_pt    : shape={codes_pt.shape} dtype={codes_pt.dtype}")

    # PyTorch encode->decode roundtrip audio.
    with torch.inference_mode():
        codes_pt_t = torch.from_numpy(codes_pt).unsqueeze(0).to(device)  # (1, CB, T)
        audio_pt   = tok.decode(codes_pt_t).audio_values.cpu().numpy().astype(np.float32)
    audio_pt = audio_pt[0, 0]  # (N,)
    print(f"[ref] audio_pt    : {audio_pt.shape[0]} samples")

    # C++ side : encode then decode. Copy the input WAV next to the script so
    # the binary auto-names its outputs here by extension swap.
    wav_tmp        = "roundtrip.wav"
    codes_cpp_path = "roundtrip.rvq"
    audio_cpp_path = "roundtrip.wav"
    shutil.copy(args.wav, wav_tmp)
    try:
        cmd_enc = [args.bin, "--model", args.model, "-i", wav_tmp]
        print(f"[cpp] {' '.join(cmd_enc)}")
        r = subprocess.run(cmd_enc, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            print(f"[cpp] encode FAILED exit={r.returncode}\n{r.stderr}", file=sys.stderr)
            sys.exit(1)

        cmd_dec = [args.bin, "--model", args.model, "-i", codes_cpp_path]
        print(f"[cpp] {' '.join(cmd_dec)}")
        r = subprocess.run(cmd_dec, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            print(f"[cpp] decode FAILED exit={r.returncode}\n{r.stderr}", file=sys.stderr)
            sys.exit(1)

        codes_cpp     = read_codes(codes_cpp_path, codes_pt.shape[0])        # (CB, T)
        audio_cpp, sr2 = sf.read(audio_cpp_path)
        if audio_cpp.ndim > 1:
            audio_cpp = audio_cpp[:, 0]
        print(f"[cpp] codes_cpp   : shape={codes_cpp.shape} dtype={codes_cpp.dtype}")
        print(f"[cpp] audio_cpp   : {audio_cpp.shape[0]} samples sr={sr2}")
    finally:
        for p in (wav_tmp, codes_cpp_path, audio_cpp_path):
            if os.path.exists(p):
                os.unlink(p)

    # Codes : exact match rate per codebook.
    print("Codes : C++ vs PyTorch reference")
    if codes_cpp.shape != codes_pt.shape:
        print(f"  WARN shape mismatch cpp={codes_cpp.shape} pt={codes_pt.shape}")
    else:
        for k in range(codes_cpp.shape[0]):
            match = float(np.mean(codes_cpp[k] == codes_pt[k]))
            print(f"  codebook {k}  match {100.0 * match:6.2f}% ({int(np.sum(codes_cpp[k] == codes_pt[k]))}/{codes_cpp.shape[1]})")
        global_match = float(np.mean(codes_cpp.flatten() == codes_pt.flatten()))
        print(f"  global   match {100.0 * global_match:6.2f}% ({int(np.sum(codes_cpp.flatten() == codes_pt.flatten()))}/{codes_cpp.size})")

    # Audio : cossim against original (lossy through RVQ) and against PyTorch
    # roundtrip (should be near 1 if our pipeline matches).
    print("Audio roundtrip : C++ encode -> C++ decode")
    stats("vs codec-ref.wav (lossy)", audio_cpp, wav.astype(np.float32))
    stats("vs PyTorch roundtrip",     audio_cpp, audio_pt)

if __name__ == "__main__":
    main()
