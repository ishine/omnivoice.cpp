# Architecture

Full technical reference for omnivoice.cpp. Companion to README.md.

## Status

Phase 0 : skeleton compilable (quantize, ggml submodule). DONE.
Phase 1 : checkpoints.sh + convert.py + quantize.sh, 2 GGUFs produits. DONE.
Phase 2 : omnivoice-codec decode-only (RVQ + fc2 + DAC). IN PROGRESS.
  Phase 2.1 : modules de base + rvq-codec.h + dac-decoder.h + dump script. DONE.
  Phase 2.2 : pipeline-codec + tools/omnivoice-codec.cpp + CMakeLists update. NEXT.
  Phase 2.3 : test cossim vs PyTorch reference. NEXT.
Phase 3 : omnivoice-codec encode (HuBERT + DAC encoder + RVQ encode). TODO.
Phase 4 : Qwen3 backbone + audio_emb + audio_heads + mask-predict (omnivoice-tts). TODO.
Phase 5 : omnivoice-tts full pipeline (auto + voice design + voice cloning). TODO.
Phase 6 : tests automatises + quantization matrix + final docs. TODO.

## Upstream model

OmniVoice by Xiaomi / k2-fsa : multilingual zero-shot TTS supporting 646 languages,
based on a diffusion-style masked language model over residual audio codes.

Single public checkpoint : `k2-fsa/OmniVoice` (Apache 2.0, 3.1 GB).
Audio tokenizer is `bosonai/higgs-audio-v2-tokenizer` (Apache 2.0), bundled
inside the OmniVoice repo as `audio_tokenizer/` subdir, sha256 identical to
the standalone mirror at `eustlb/higgs-audio-v2-tokenizer`.

Backbone : Qwen3 0.6B with custom audio IO (no text lm_head).
Audio tokenizer : DAC acoustic codec + HuBERT semantic + 8 RVQ codebooks, fused.
Sample rate : 24 kHz mono. Frame rate : 25 Hz. Hop length : 960 samples.

## Build

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/omnivoice.cpp.git
cd omnivoice.cpp
./buildcuda.sh      # NVIDIA GPU
./buildvulkan.sh    # AMD/Intel GPU
./buildcpu.sh       # CPU only
./buildall.sh       # all backends, runtime DL loading
```

GGML submodule : `https://github.com/ServeurpersoCom/ggml.git` (fork with
`GGML_OP_SNAKE` and `GGML_OP_COL2IM_1D` custom ops, CPU/CUDA/Metal/Vulkan).

## Model conversion

```
./checkpoints.sh         # hf download k2-fsa/OmniVoice -> checkpoints/OmniVoice/  (3.1 GB)
./convert.py             # 2 GGUFs in BF16 -> models/
./quantize.sh            # base LM Q8_0 (tokenizer stays BF16)
```

Outputs :
```
models/omnivoice-base-BF16.gguf       1.2 GB    LLM + audio_emb + audio_heads + tokenizer
models/omnivoice-base-Q8_0.gguf       626 MB    quantized base, 1.9x compression
models/omnivoice-tokenizer-F32.gguf   702 MB    HuBERT + DAC + RVQ + fc/fc2 (native F32)
```

Tokenizer GGUF preserves the source dtype 1:1. The reference checkpoint
audio_tokenizer/model.safetensors is integrally F32, so the GGUF stays F32 :
no precision is invented or destroyed in the conversion. The RVQ residual
chain accumulates rounding noise across 8 codebooks and any BF16 truncation
upstream pulls the late codebooks below 50% match against the reference.

Quantization policy : Q8_0 only. The base LM is small (612M params), lower
quants do not pay off in quality versus Q8_0.

## GGUF layout

`omnivoice-base-{quant}.gguf` (arch = `omnivoice-lm`) :
```
metadata
  general.architecture                   omnivoice-lm
  block_count                            28
  embedding_length                       1024
  feed_forward_length                    3072
  head_count                             16
  head_count_kv                          8        (GQA 2:1)
  key_length                             128
  vocab_size                             151676
  context_length                         40960
  layer_norm_rms_eps                     1e-6
  rope_freq_base                         1e6
  omnivoice.tie_word_embeddings          true
  omnivoice.num_audio_codebook           8
  omnivoice.audio_vocab_size             1025
  omnivoice.audio_mask_id                1024
  omnivoice.audio_codebook_weights       [8, 8, 6, 6, 4, 4, 2, 2]
  omnivoice.special.denoise              151669
  omnivoice.special.lang_start           151670
  omnivoice.special.lang_end             151671
  omnivoice.special.instruct_start       151672
  omnivoice.special.instruct_end         151673
  omnivoice.special.text_start           151674
  omnivoice.special.text_end             151675
  tokenizer (Qwen2 BPE, 151676 vocab, 151387 merges, 33 added_tokens)

tensors (312)
  llm.embed_tokens.weight                (151676, 1024)
  llm.norm.weight                        (1024,)
  llm.layers.0..27.{q,k,v,o}_proj.weight                              GQA, no bias
  llm.layers.0..27.self_attn.{q_norm, k_norm}.weight                  per-head RMSNorm (128,)
  llm.layers.0..27.{input,post_attention}_layernorm.weight            RMSNorm
  llm.layers.0..27.mlp.{gate,up,down}_proj.weight                     SwiGLU, no bias
  audio_embeddings.weight                (8200, 1024)                 8 codebooks * 1025 vocab
  audio_heads.weight                     (8200, 1024)                 audio output, no bias

skipped (recomputable)
  codebook_layer_offsets                 = arange(8) * 1025
```

`omnivoice-tokenizer-{quant}.gguf` (arch = `omnivoice-tokenizer`) :
```
metadata
  omnivoice.sample_rate                  24000
  omnivoice.semantic_sample_rate         16000
  omnivoice.downsample_factor            320
  omnivoice.codebook_size                1024
  omnivoice.codebook_dim                 64
  omnivoice.acoustic.encoder_hidden_size 64
  omnivoice.acoustic.decoder_hidden_size 1024
  omnivoice.acoustic.hidden_size         256
  omnivoice.acoustic.n_codebooks         9        (only 8 used)
  omnivoice.acoustic.hop_length          960
  omnivoice.acoustic.upsampling_ratios   [8, 5, 4, 2, 3]
  omnivoice.acoustic.downsampling_ratios [8, 5, 4, 2, 3]
  omnivoice.semantic.hidden_size         768       (HuBERT base)
  omnivoice.semantic.intermediate_size   3072
  omnivoice.semantic.num_attention_heads 12
  omnivoice.semantic.num_hidden_layers   12
  omnivoice.semantic.num_feat_extract_layers 7
  omnivoice.semantic.conv_dim            [512]*7
  omnivoice.semantic.conv_kernel         [10, 3, 3, 3, 3, 2, 2]
  omnivoice.semantic.conv_stride         [5, 2, 2, 2, 2, 2, 2]
  omnivoice.semantic.num_conv_pos_embeddings        128
  omnivoice.semantic.num_conv_pos_embedding_groups  16
  omnivoice.semantic.layer_norm_eps      1e-5

tensors (486)
  acoustic_encoder.*                     DAC encoder, 5 blocks, downsamples 8 5 4 2 3
  acoustic_decoder.*                     DAC decoder, 5 blocks, upsamples 8 5 4 2 3
  encoder_semantic.*                     semantic conv blocks, encode side
  semantic_model.*                       HuBERT base, weight_norm folded
  quantizer.quantizers.0..7.{codebook.embed, project_in.{w,b}, project_out.{w,b}}
  fc.{weight, bias}                      1024 -> 1024 (after concat acoustic + semantic)
  fc2.{weight, bias}                     1024 -> 256 (before DAC decoder)

skipped (training-only)
  decoder_semantic.*                     auxiliary HuBERT-feature reconstruction loss
  fc1.{weight, bias}                     auxiliary loss path
  quantizer.*.codebook.{cluster_size, embed_avg, inited}    RVQ EMA buffers
  parametrizations.weight.original{0,1}  folded into pos_conv weight
```

Single weight_norm fold at convert time : `semantic_model.encoder.pos_conv_embed.conv.weight`,
formula `weight = v * g / ||v||_{dim=(0,1)}` matching `torch._weight_norm(v, g, dim=2)`.
Validated bit-perfect, max abs diff 3.9e-7 vs PyTorch reference.

## Component architecture

### Qwen3 0.6B backbone (custom IO)

Standard Qwen3 modulo two changes :
- input embed : hybrid text + audio, weighted sum across 8 codebooks gated by `audio_mask`
- output : custom `audio_heads` Linear (8200, 1024), no text `lm_head`

```
input_ids [B, 8, S] int          (text on row 0, audio codes on rows 1..7)
audio_mask [B, S] bool

text_emb  = embed_tokens(input_ids[:, 0, :])               (B, S, 1024)
shifted   = input_ids * audio_mask + offsets[None, :, None] (B, 8, S)
                                                            offsets = arange(8) * 1025
audio_emb = audio_embeddings(shifted).sum(dim=1)           (B, S, 1024)
inputs    = where(audio_mask, audio_emb, text_emb)         (B, S, 1024)

x = qwen3_forward(inputs, attention_mask, position_ids)    (B, S, 1024)

logits_flat = x @ audio_heads.weight.T                     (B, S, 8200)
logits      = reshape (B, 8, S, 1025)
```

Qwen3 specifics already in llama.cpp :
- 28 layers, hidden 1024, intermediate 3072
- 16 query heads + 8 KV heads (GQA 2:1), head_dim 128
- per-head RMSNorm on Q and K (q_norm, k_norm shape (128,)) before RoPE
- no bias on Q/K/V/O/MLP
- RoPE theta = 1e6
- SwiGLU MLP
- tie_word_embeddings = true (but `lm_head` absent in checkpoint, output is audio_heads only)

### Mask-predict generation (no KV cache)

Prompt (per item, broadcast across 8 codebooks) :
```
[<|denoise|>]?
<|lang_start|> {iso_code or "None"} <|lang_end|>
<|instruct_start|> {style or "None"} <|instruct_end|>
<|text_start|> {ref_text + " " + text} <|text_end|>
{ref_audio_codes}?
{MASK x num_target_tokens}
```

Unconditional prompt for CFG = the last `num_target_tokens` mask tokens only.
Batched (cond + uncond) doubles the batch dim.

```
for step in 0..num_step-1 :
    forward(input_ids, audio_mask, attention_mask)         (2B, 8, S, 1025)
    log_probs = logsoftmax(c + cfg_scale * (c - u))
    log_probs[..., MASK_ID] = -inf
    if class_temp > 0 :
        keep_top_k_ratio(log_probs, 0.1)
        gumbel_sample(temp = class_temp)
    pred  = argmax(log_probs)
    score = log_probs.max - layer_idx * layer_penalty      (5.0)
    if pos_temp > 0 :
        score += gumbel * pos_temp
    score[already_unmasked] = -inf
    topk_idx = topk(score.flatten(), schedule[step])
    tokens[topk_idx] = pred[topk_idx]
    update batch_input_ids cond and uncond
```

Schedule of newly unmasked positions per step is computed from
`_get_time_steps(t_start=0, t_end=1, num_step, t_shift)` then
`ceil(N_total * (t[step+1] - t[step]))`. 32 steps default.

KV cache is not usable : tokens reveal in arbitrary positions across 8 codebook
layers each step, so each step is a full prefill. Inference is therefore
`num_step * 2 * forward_full(B, S)` (the 2 accounts for cond + uncond).

### Audio tokenizer pipeline

Encode (voice cloning) :
```
ref_audio @ 24 kHz                                  (1, 1, T_samples)
  -> resample 16 kHz                                (kaiser polyphase)
  -> HuBERT semantic_model (12 transformer layers)  output 13 hidden states
  -> mean over the 13 hidden states                 (1, 768, T_sem)
  -> downsample by 2                                (semantic_downsample_factor)
  -> SemanticEncoder (conv blocks)                  (1, 768, T_frames)
  -> e_acoustic (DAC encoder, 5 down-blocks)        (1, 256, T_frames)
  -> concat dim=1                                   (1, 1024, T_frames)
  -> fc Linear (1024 -> 1024)                       (1, 1024, T_frames)
  -> RVQ encode (8 codebooks residual)              (1, 8, T_frames) int @ 25 fps
```

Decode (TTS path, what omnivoice.cpp tackles first) :
```
codes [B, 8, T] int
  -> RVQ decode :
     for k in 0..7 :
         e_k = codebook[k].embed[codes[k, :]]        (B, T, 64)
         p_k = e_k @ project_out[k].W.T + bias[k]    (B, T, 1024)
         out += p_k
  -> transpose (B, 1024, T)
  -> fc2 Linear (1024 -> 256)                        (B, 256, T)
  -> acoustic_decoder DAC :
     conv1 (256 -> 1024, k=7, pad=3)
     for block in 0..4, ratios [8, 5, 4, 2, 3] :
         snake1 (alpha)
         conv_t1 (IC -> OC, k=2*r, stride=r,
                  padding=ceil(r/2), output_padding=r%2)
         for res_unit in 0..2, dilations [1, 3, 9] :
             snake1 (alpha)
             conv1 (OC, OC, k=7, dil=d, pad=3*d)
             snake2 (alpha)
             conv2 (OC, OC, k=1)
             residual add
     snake1 (alpha) final
     conv2 (32 -> 1, k=7, pad=3)
  -> audio (B, 1, 960*T)
```

960x upsample = 8 * 5 * 4 * 2 * 3. T_in @ 25 fps -> T_out @ 24 kHz exact.

### DAC decoder block channels

```
block 0 : IC=1024  OC=512  stride=8  K=16  pad=4  output_pad=0
block 1 : IC=512   OC=256  stride=5  K=10  pad=3  output_pad=1
block 2 : IC=256   OC=128  stride=4  K=8   pad=2  output_pad=0
block 3 : IC=128   OC=64   stride=2  K=4   pad=1  output_pad=0
block 4 : IC=64    OC=32   stride=3  K=6   pad=2  output_pad=1
final   : 32 -> 1
```

PyTorch ConvTranspose1d formula :
`T_out = (T_in - 1)*stride - 2*padding + dilation*(kernel - 1) + output_padding + 1`

With our params (d=1, k=2*s, p=ceil(s/2), op=s%2) the formula collapses to
`T_out = stride * T_in` exactly for all five blocks.

### Snake activation

DAC HF formula (`Snake1d.forward`) :
`y = x + (alpha + 1e-9).reciprocal() * sin(alpha * x)^2`

`ggml_snake(x, a, inv_b)` computes `y = x + sin^2(a * x) * inv_b`. Mapping :
- `a = alpha`             (loaded direct, BF16 to F32)
- `inv_b = 1/(alpha + 1e-9)`  (precomputed CPU side at load, F32)

Both stored as F32 `[1, C]` tensors.

`alpha` shape in checkpoint : `(1, C, 1)`, ggml ne = (1, C, 1). C lives on
ne[1]. Loader reads C from `mt->ne[1]`.

### ConvTranspose1d via GEMM + col2im_1d

PyTorch `nn.ConvTranspose1d(IC, OC, kernel=K, stride=s, padding=p)` with weight
shape `(IC, OC, K)`. GGML decomposition :

```
1. Permute weight (IC, OC, K) PyTorch -> (IC, K*OC) ggml at load time.
   Layout : dst[(oc*K + k) * IC + ic] = src[ic*OC*K + oc*K + k]
   This makes k vary faster than oc inside the K*OC axis, matching what
   ggml_compute_forward_col2im_1d_impl expects :
     col_data[(oc * K + k) + t_in * K_OC]

2. Build runtime graph :
   xt   = ggml_cont(ctx, ggml_transpose(ctx, x))           # [IC, T_in]
   col  = ggml_mul_mat(ctx, w, xt)                         # [K*OC, T_in]
   y    = ggml_col2im_1d(ctx, col, stride, OC, padding)    # [T_no_op, OC]
   if (output_pad > 0)
       y = ggml_pad(ctx, y, output_pad, 0, 0, 0)           # right-pad zeros
   if (bias)
       y = ggml_add(ctx, y, bias_2d)
```

Validated math : `T_no_op = (T_in - 1)*stride + K - 2*pad`. Adding `output_pad`
right-pad gives the PyTorch output size exactly.

### RVQ codec

Per-codebook tensors (k = 0..7) :
```
codebook.embed         (1024, 64) PyTorch -> ggml ne=(64, 1024)
project_in.weight      (64, 1024) PyTorch -> ggml ne=(1024, 64)   encode-only
project_in.bias        (64,)                                       encode-only
project_out.weight     (1024, 64) PyTorch -> ggml ne=(64, 1024)
project_out.bias       (1024,)
```

Decode graph (per codebook k, accumulated) :
```
codes_k = ggml_view_1d(codes, T, k * stride)               # [T] i32
e_k     = ggml_get_rows(embed[k], codes_k)                 # [64, T]
p_k     = ggml_mul_mat(project_out_w[k], e_k)              # [1024, T]
p_k     = ggml_add(p_k, project_out_b[k])
acc    += p_k
```

Encode (residual, deferred to phase 3) :
```
residual = embeddings_in
for k in 0..7 :
    e_k       = project_in[k](residual)
    codes_k   = argmin_i ||e_k - codebook[k].embed[i]||^2
    quantized = project_out[k](codebook[k].embed[codes_k])
    residual -= quantized
```

### HuBERT semantic encoder (phase 3+)

12 transformer layers Pre-LN, GELU FFN, MHA 12 heads * 64 dim, biases on all
QKVO. Pre-conv feature extractor : 7 Conv1D layers, kernels `[10, 3, 3, 3, 3, 2, 2]`,
strides `[5, 2, 2, 2, 2, 2, 2]`, GroupNorm on first only, GELU between.
Feature projection LayerNorm + Linear (512 -> 768). Positional embedding via
grouped Conv1D (128 kernel, 16 groups), `weight_norm` folded at convert time.
Final LayerNorm.

Output computation :
```
mean(stack(all_13_hidden_states, dim=1), dim=1)            # (B, T_sem, 768)
```

This is unusual : averages across initial input + 12 transformer layer
outputs, not just the last hidden state.

### Voice clone reference encoding (phase 3)

```
ref_audio @ 24k   -> resample 16k         (kaiser polyphase, libsamplerate)
                  -> pad 160 each side
                  -> HuBERT.feature_extractor (320x downsample)
                  -> HuBERT.feature_projection (LayerNorm + Linear)
                  -> + pos_conv_embed (folded)
                  -> 12 transformer layers
                  -> mean over 13 hidden states
                  -> [::semantic_downsample_factor=2]
                  -> SemanticEncoder convs
                  -> e_semantic (B, 768, T_frames)
ref_audio @ 24k   -> DAC acoustic_encoder (5 down-blocks, ratios 8 5 4 2 3)
                  -> e_acoustic (B, 256, T_frames)
concat dim=1 -> (B, 1024, T_frames)
fc Linear -> (B, 1024, T_frames)
RVQ encode -> codes (B, 8, T_frames) int @ 25 fps
```

## Module map

```
src/
  backend.h            (acestep copy)  GGML backend init, sched, env override
  weight-ctx.h         (acestep copy)  generic weight context for GGUF loaders
  gguf-weights.h       (acestep copy)  mmap GGUF, gf_load_tensor, gf_get_*
  audio-io.h           (acestep base)  WAV read + mono write (S16/S24/F32)
  audio-resample.h     (acestep copy)  kaiser polyphase 24k <-> 16k
  wav.h                (acestep copy)  WAV header reader (PCM16/24/F32, mono/stereo)

  rvq-codec.h          (NEW)           RVQ decode (8 codebooks, lookup + project_out + sum)
  dac-decoder.h        (NEW)           DAC decoder (5 blocks Snake + ConvTranspose1d + 3 res_units)
  duration-estimator.h (NEW)           byte-perfect port of RuleDurationEstimator (sec -> tokens)

  TODO :
  pipeline-codec.h/.cpp                fc2 + RVQ + DAC + WAV write
  bpe.h                                Qwen2 BPE tokenizer (reuse acestep style)
  qwen3-lm.h                           Qwen3 backbone with audio_emb / audio_heads
  prompt.h                             prompt builder (denoise + lang + instruct + text + mask)
  mask-predict.h                       32-step iterative loop, CFG batched, layer penalty, gumbel
  nonverbal-tags.h                     13 tags split tokenization
  voice-design.h                       instruct mapping EN <-> ZH
  lang-map.h                           646 ISO 639-3 codes generated from docs/lang_id_name_map.tsv
  hubert-enc.h                         HuBERT base (feature_extractor + pos_conv + 12 layers + final_norm)
  semantic-enc.h                       encoder_semantic conv blocks
  dac-encoder.h                        DAC encoder (mirror of decoder)
  audio-tokenizer.h                    combine HuBERT + DAC enc/dec + RVQ + fc/fc1/fc2
  pipeline-tts.h/.cpp                  full TTS (auto + design + clone)

tools/
  quantize.cpp         (acestep adapt) GGUF requantizer
  version.cmake        (acestep adapt) git hash -> OMNIVOICE_VERSION

  TODO :
  omnivoice-codec.cpp                     CLI : codes <-> WAV
  omnivoice-tts.cpp                       CLI : text -> WAV (auto / design / clone modes)

tests/
  dump-codec-ref.py    (NEW)           PyTorch reference : codes + WAV from "Hello world"
  CMakeLists.txt                       empty placeholder, fills as tests land
  TODO :
  test-rvq-roundtrip.cpp               encode + decode RVQ bit-identical
  test-dac-cossim.cpp                  DAC vs PyTorch cossim > 0.999
  test-prompt.cpp                      prompt builder byte-perfect vs Python
  debug-decoder-cossim.py              capture intermediates for cross-validation
```

## Coding conventions (Pascal style, Georgi inspired)

### Source files

C++17 only. Headers `.h` with implementation inline as `static` functions
(no separate `.cpp` for self-contained modules). Implementation files reserved
for orchestration that pulls multiple modules together (`pipeline-*.cpp`).

Each header starts with :
```
// filename : one-line description
// optional second/third line for context
#pragma once

#include "..."

#include <...>
```

No banners (no `===` or `###` decorations). Comments describe the present, never
the past or evolution. No legacy fallbacks. If broken code can be removed
without breaking the build, it was superfluous.

KISS over cleverness. Inline algorithms when they fit on the screen. Minimal
state, no class hierarchies for plain data. Structs hold tensors and config,
free functions operate on them.

### Naming

Types `CamelCase` : `DACDecoder`, `RVQCodec`, `Qwen3LM`, `WeightCtx`.
Functions and variables `snake_case` with module prefix : `dac_load`,
`dac_build_graph`, `rvq_decode_graph`, `gf_load_tensor`, `wctx_alloc`.
File-private constants `UPPER_SNAKE` macros : `DAC_NUM_BLOCKS`, `RVQ_NUM_CODEBOOKS`.

### ASCII

Code and comments are pure ASCII. No em dashes, no minus signs in prose
(arrow `->` is OK in technical comments and pseudocode), no curly quotes,
no bullet points in source comments.

### GGML tensor conventions

PyTorch shape `(out, in)` for a Linear weight stores as ggml `ne[0]=in,
ne[1]=out`. The GGUF tensor-shape array is reversed, so reading
`reversed(t.shape)` from gguf-py gives the PyTorch shape directly.

For PyTorch Conv1d weight `(OC, IC, K)`, ggml ne = `(K, IC, OC)`. The kernel
axis is innermost (contiguous in memory).

For PyTorch ConvTranspose1d weight `(IC, OC, K)`, the convert-time permutation
to ggml `(IC, K*OC)` rearranges `(oc*K + k) * IC + ic` so col2im_1d gets the
correct column matrix.

`ggml_mul_mat(A, B)` : with A.ne[0] = K (must match B.ne[0]), A.ne[1] = M,
B.ne[1] = N, output has ne = (N, M). In PyTorch terms, A is `(M, K)`,
B is `(N, K)`, output is `(M, N)`, which equals `A @ B^T`.

### Weight loading pattern (acestep VAE style)

For modules with at-load transforms (snake reciprocal, conv-transpose
permutation), bypass `WeightCtx` and follow the 3-phase pattern :

```
// Phase 1 : describe tensors (no_alloc context)
struct ggml_init_params p = { ctx_size, NULL, true };
m->weight_ctx = ggml_init(p);
m->c1w        = ggml_new_tensor_3d(ctx, GGML_TYPE_BF16, K, IC, OC);
... (all tensors)

// Phase 2 : allocate one big backend buffer for all of them
m->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
ggml_backend_buffer_set_usage(m->weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

// Phase 3 : copy data per tensor with the appropriate transform
load_bf16(m->c1w, gf, "...");           // raw passthrough
load_bias_f32(m->c1b, gf, "...");       // BF16 -> F32 cast
load_alpha(s->a, gf, "...", false);     // BF16 alpha -> F32
load_alpha(s->inv_b, gf, "...", true);  // 1 / (alpha + 1e-9)
load_ctw(m->ctw, gf, "...");            // permute (IC, OC, K) -> (IC, K*OC)
```

For modules with pure passthrough loads (LM, simple convs), use `WeightCtx`
and `gf_load_tensor` for cleaner code.

### Backend init

Always `backend_init("MOD")` then `backend_sched_new(bp, max_nodes)`.
Backend cache is shared across modules in the same binary, refcounted.
GPU backend used by default, CPU backend kept as scheduler fallback.

### Reference conventions from acestep.cpp

Mirror its module decomposition where the math is identical : see
`src/vae.h` for the upsample 5-block + Snake + ConvTranspose1d + 3 res_units
template, which is structurally what OmniVoice DAC uses.

Differences vs acestep VAE Oobleck :
- Snake : DAC HF stores `alpha` direct, OmniVoice loads `inv_b = 1/(alpha + 1e-9)`.
  Acestep VAE Oobleck uses log-reparametrized `alpha, beta`, loads
  `a = exp(alpha)`, `inv_b = 1/exp(beta)`.
- Weight_norm : OmniVoice DAC has weights already merged in checkpoint
  (only HuBERT pos_conv has parametrizations, folded at convert).
  Acestep VAE Oobleck has all convs in `weight_g + weight_v`, folded at load.
- ConvTranspose1d : OmniVoice has `output_padding = stride % 2` (handled via
  `ggml_pad` after `col2im_1d`). Acestep has output_padding = 0.

## Custom GGML ops used

Provided by the ServeurpersoCom/ggml fork :

`ggml_snake(ctx, x, a, inv_b)` : `y = x + sin^2(a * x) * inv_b`.
F32 / F16 / BF16 input/output. CPU + CUDA + Metal + Vulkan.

`ggml_col2im_1d(ctx, a, s0, oc, p0)` : scatter-add `[K*OC, T_in]` columns into
`[T_out, OC]` signal where `T_out = (T_in - 1)*s0 + K - 2*p0`. Layout requires
k to vary faster than oc inside the K*OC axis. F32 / F16 / BF16. All backends.

`ggml_pad(ctx, a, p0, p1, p2, p3)` : right-pad with zeros along each axis.
Used for PyTorch ConvTranspose1d `output_padding`.

## What omnivoice.cpp is NOT doing

- Whisper ASR for auto-transcription of voice-clone reference. The user must
  provide `ref_text`. If we want auto-transcribe later, we plug whisper.cpp
  rather than re-port Whisper.
- Evaluation models (WER, SIM-o, UTMOS, paraformer, hubert-large-ls960-ft,
  utmos22, wavlm). These are inference-irrelevant.
- Fine-tuning / training. Inference only.
- ComfyUI / Gradio integration. CLI binaries only.

## Glossary

RVQ : Residual Vector Quantization. Stack of codebooks where each one quantizes
the residual from the previous codebook's reconstruction.

DAC : Descript Audio Codec. Convolutional encoder/decoder over residual VQ
codes, originally Latent Diffusion Speech Codec by Descript.

HuBERT : Hidden-Unit BERT. Transformer encoder pretrained with masked acoustic
unit prediction. Used here to extract semantic embeddings from raw audio.

Snake : `y = x + (1/alpha) * sin^2(alpha * x)`, a periodic activation
introduced in BigVGAN, replaces LeakyReLU in the DAC encoder/decoder.

CFG : Classifier-Free Guidance. Generation trick where the model is run twice
(conditional and unconditional) and outputs combined as
`c + scale * (c - u)` to amplify the conditional signal.

Mask-predict / MaskGIT : iterative non-autoregressive decoding where masked
tokens are progressively unmasked over a fixed number of steps, prioritizing
high-confidence positions per step.
