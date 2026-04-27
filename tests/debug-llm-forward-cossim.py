#!/usr/bin/env python3
# debug-llm-forward-cossim.py: validate the C++ full LLM forward chain
# (custom embed -> 28L stack -> audio_heads) end-to-end against the Python
# reference. Both sides receive the exact same synthetic input_ids and
# audio_mask. The Python reference logits are computed in memory; the C++
# logits come from omnivoice-tts --llm-test reading a temp input dump.

import os
import struct
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, "../OmniVoice")
from omnivoice.models.omnivoice import OmniVoice

BIN            = "../build/omnivoice-tts"
MODEL_LM       = "../models/omnivoice-base-F32.gguf"
CHECKPOINT_DIR = "../checkpoints/OmniVoice"

S    = 64
SEED = 42

def write_input_dump(path, input_ids_ks, audio_mask_s):
    K, Slen = input_ids_ks.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", K, Slen))
        f.write(np.ascontiguousarray(input_ids_ks, dtype="<i4").tobytes())
        f.write(np.ascontiguousarray(audio_mask_s, dtype="<i4").tobytes())

def read_logits_dump(path):
    with open(path, "rb") as f:
        V, K, Slen = struct.unpack("<iii", f.read(12))
        n = V * K * Slen
        data = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(Slen, K, V)
    return V, K, Slen, data

def build_synthetic_inputs(model):
    K          = model.config.num_audio_codebook
    V          = model.config.audio_vocab_size
    vocab_text = model.config.llm_config.vocab_size
    device     = next(model.parameters()).device

    torch.manual_seed(SEED)
    text_part   = S // 2
    audio_part  = S - text_part
    text_row    = torch.randint(0, vocab_text, (text_part,), device=device, dtype=torch.long)
    audio_codes = torch.randint(0, V, (K, audio_part), device=device, dtype=torch.long)

    input_ids = torch.empty((1, K, S), dtype=torch.long, device=device)
    input_ids[0, :, :text_part] = text_row.unsqueeze(0).expand(K, text_part)
    input_ids[0, :, text_part:] = audio_codes

    audio_mask = torch.zeros((1, S), dtype=torch.bool, device=device)
    audio_mask[0, text_part:] = True

    return input_ids, audio_mask, K, V

def reference_logits(model, input_ids, audio_mask):
    device         = next(model.parameters()).device
    attention_mask = torch.ones((1, 1, S, S), dtype=torch.bool, device=device)
    position_ids   = torch.arange(S, device=device, dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        inputs_embeds = model._prepare_embed_inputs(input_ids, audio_mask)
        llm_outputs   = model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        hidden       = llm_outputs[0]
        logits_flat  = model.audio_heads(hidden)
        audio_logits = logits_flat.view(1, S, -1, model.config.audio_vocab_size)
    return audio_logits.detach().cpu().numpy()[0].copy()

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

    input_ids, audio_mask, K, V = build_synthetic_inputs(model)
    print("config: K=%d V=%d S=%d" % (K, V, S))

    print("computing reference logits ...")
    ref = reference_logits(model, input_ids, audio_mask)
    print("ref logits shape=%s range=[%.4f, %.4f] std=%.4f" %
          (ref.shape, ref.min(), ref.max(), ref.std()))

    ids_ks = input_ids.detach().cpu().numpy()[0].astype("int32")
    mask_s = audio_mask.detach().cpu().numpy()[0].astype("int32")

    input_path = "llm_forward_input.bin"
    cpp_path   = "llm_forward_cpp.bin"
    try:
        write_input_dump(input_path, ids_ks, mask_s)
        print("running C++ full LM forward ...")
        rc = subprocess.run(
            [BIN, "--model", MODEL_LM, "--llm-test", input_path, "-o", cpp_path],
            check=False,
        ).returncode
        if rc != 0:
            print("error: omnivoice-tts returned %d" % rc, file=sys.stderr)
            sys.exit(1)

        Vc, Kc, Sc, cpp = read_logits_dump(cpp_path)
        if (V, K, S) != (Vc, Kc, Sc):
            print("error: shape mismatch ref=(V=%d K=%d S=%d) cpp=(V=%d K=%d S=%d)" %
                  (V, K, S, Vc, Kc, Sc), file=sys.stderr)
            sys.exit(1)

        print("cpp range=[%.4f, %.4f] std=%.4f" % (cpp.min(), cpp.max(), cpp.std()))
        abs_diff = np.abs(cpp - ref)
        print("abs diff   max=%.6f  mean=%.6f  p99=%.6f" %
              (abs_diff.max(), abs_diff.mean(), np.percentile(abs_diff, 99)))

        print("per codebook cossim:")
        for k in range(K):
            ref_k = ref[:, k, :].reshape(-1)
            cpp_k = cpp[:, k, :].reshape(-1)
            cs_k  = float((ref_k * cpp_k).sum() /
                          max(np.linalg.norm(ref_k) * np.linalg.norm(cpp_k), 1e-12))
            print("  k=%d  cossim=%.6f" % (k, cs_k))

        flat_cs = float((ref.reshape(-1) * cpp.reshape(-1)).sum() /
                        max(np.linalg.norm(ref) * np.linalg.norm(cpp), 1e-12))
        print("cossim global = %.6f" % flat_cs)

        ref_argmax = ref.argmax(axis=2)
        cpp_argmax = cpp.argmax(axis=2)
        agree = (ref_argmax == cpp_argmax).mean()
        print("argmax agreement = %.4f  (%d / %d slots)" %
              (agree, int((ref_argmax == cpp_argmax).sum()), S * K))
    finally:
        for p in (input_path, cpp_path):
            if os.path.exists(p):
                os.unlink(p)

if __name__ == "__main__":
    main()
