# omnivoice.cpp

Local AI text-to-speech with voice cloning, powered by GGML.
C++17 port of OmniVoice (k2-fsa/OmniVoice). 646 languages, voice cloning,
voice design. Runs on CPU, CUDA, ROCm, Metal, Vulkan.

Status: skeleton, work in progress.

## Build

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/omnivoice.cpp.git
cd omnivoice.cpp
./buildcuda.sh        # NVIDIA GPU
./buildvulkan.sh      # AMD/Intel GPU (Vulkan)
./buildcpu.sh         # CPU only
./buildall.sh         # all backends, runtime DL loading
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full technical reference.

## License

MIT. See [LICENSE](LICENSE).

Upstream model: OmniVoice by Xiaomi / k2-fsa, Apache-2.0.
