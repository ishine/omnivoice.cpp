#!/usr/bin/env python3
# debug-maskgit-cossim.py: validate maskgit_generate against the reference
# _generate_iterative. Compares audio_tokens bytewise with both temperatures
# forced to zero on each side (pure argmax + pure top-k confidence).
#
# This test reports bytewise mismatch as a fail to expose the structural
# divergence at the LLM forward argmax level (cf. debug-llm-forward-cossim.py).
# Reaching bytewise match requires solving that divergence first.

import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, "../OmniVoice")
from omnivoice.models.omnivoice import (
    GenerationTask,
    OmniVoice,
    OmniVoiceGenerationConfig,
)

BIN            = "../build/omnivoice-tts"
MODEL_LM       = "../models/omnivoice-base-F32.gguf"
CHECKPOINT_DIR = "../checkpoints/OmniVoice"

# Short cases to keep the reference run reasonable. The decoder runs 32 LLM
# forwards with B'=2 batched, scaling roughly linearly with T.
CASES = [
    {"text": "Hello world.", "lang": "English", "instruct": "Speak softly",
     "duration": 8, "denoise": False},
    {"text": "Voici une phrase.", "lang": "French", "instruct": "Calmly",
     "duration": 12, "denoise": False},
]

def read_dump(path, K):
    data = np.fromfile(path, dtype="<i4")
    T    = int(data.size // K)
    return K, T, data.reshape(K, T)

def run_py(model, case, out_path):
    K = model.config.num_audio_codebook
    T = case["duration"]
    task = GenerationTask(
        batch_size=1,
        texts=[case["text"]],
        target_lens=[T],
        langs=[case["lang"]],
        instructs=[case["instruct"]],
        ref_texts=[None],
        ref_audio_tokens=[None],
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
    tokens_np = tokens.cpu().numpy().astype("int32")
    np.ascontiguousarray(tokens_np).tofile(out_path)

def run_cpp(case, out_path):
    args = [BIN, "--model", MODEL_LM, "--maskgit-test",
            "--duration", str(case["duration"]),
            "-o", out_path]
    if case["lang"] is not None:
        args += ["--lang", case["lang"]]
    if case["instruct"] is not None:
        args += ["--instruct", case["instruct"]]
    if not case["denoise"]:
        args += ["--no-denoise"]
    return subprocess.run(args, input=case["text"], text=True,
                          check=False, capture_output=False).returncode

def report(toks_c, toks_p):
    K, T = toks_c.shape
    total = K * T
    diff = (toks_c != toks_p)
    n_diff = int(diff.sum())
    print("  total=%d  matching=%d  diff=%d  agreement=%.4f" %
          (total, total - n_diff, n_diff, 1.0 - n_diff / total))
    for k in range(K):
        n_k = int(diff[k].sum())
        agree_k = 1.0 - n_k / T
        print("  k=%d  diff=%d  agreement=%.4f" % (k, n_k, agree_k))
    if n_diff:
        idx = np.argwhere(diff)
        first = idx[0]
        k0, t0 = int(first[0]), int(first[1])
        print("  first diff at (k=%d, t=%d)  cpp=%d  py=%d" %
              (k0, t0, int(toks_c[k0, t0]), int(toks_p[k0, t0])))

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
        cpp_path = "maskgit_cpp.bin"
        py_path  = "maskgit_py.bin"
        try:
            print("[%d] running cpp ..." % i)
            rc_cpp = run_cpp(case, cpp_path)
            if rc_cpp != 0:
                print("[%d] CRASH cpp=%d  case=%s" % (i, rc_cpp, case))
                n_fail += 1
                continue
            print("[%d] running py ..." % i)
            run_py(model, case, py_path)
            K_model = model.config.num_audio_codebook
            K_c, T_c, toks_c = read_dump(cpp_path, K_model)
            K_p, T_p, toks_p = read_dump(py_path,  K_model)
            if (K_c, T_c) != (K_p, T_p):
                print("[%d] SHAPE MISMATCH cpp=(%d,%d) py=(%d,%d)" %
                      (i, K_c, T_c, K_p, T_p))
                n_fail += 1
                continue
            print("[%d] case=%s K=%d T=%d" % (i, case, K_c, T_c))
            same = np.array_equal(toks_c, toks_p)
            report(toks_c, toks_p)
            if same:
                print("[%d] OK   bytewise match" % i)
                n_pass += 1
            else:
                print("[%d] FAIL bytewise mismatch" % i)
                n_fail += 1
        finally:
            for p in (cpp_path, py_path):
                if os.path.exists(p):
                    os.unlink(p)

    print("[Summary] %d pass / %d fail / %d total" % (n_pass, n_fail, len(CASES)))
    sys.exit(0 if n_fail == 0 else 1)

if __name__ == "__main__":
    main()
