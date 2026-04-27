#!/bin/bash
# Derive lighter GGUFs from the F32 source-of-truth produced by convert.py.
#   omnivoice-base-F32.gguf       -> -BF16.gguf (CUDA), -Q8_0.gguf (CPU)
#   omnivoice-tokenizer-F32.gguf  -> -BF16.gguf only.
# The tokenizer is never quantized to Q8_0 : the DAC decoder convolutions and
# the RVQ codebooks are the audio output path and need full precision. Same
# policy as acestep.cpp keeping the VAE in BF16.

set -eu

Q="./build/quantize"

quantize() {
    local src="$1" type="$2"
    local out="${src/-F32.gguf/-${type}.gguf}"
    if [ -f "$out" ]; then
        echo "[Skip] $out"
    else
        $Q "$src" "$out" "$type"
    fi
}

quantize models/omnivoice-base-F32.gguf BF16
quantize models/omnivoice-base-F32.gguf Q8_0

quantize models/omnivoice-tokenizer-F32.gguf BF16
quantize models/omnivoice-tokenizer-F32.gguf Q8_0
