# Changelog

## [Unreleased]

### Fixed
- **Docker Deployment**: Fixed critical `VOICES_DIR` path issue in `docker-env.example` that prevented containers from starting due to invalid volume mounting.
- **Dependency Installation**: Updated `Dockerfile` to remove global `PIP_ONLY_BINARY` restriction, allowing `langdetect` (required for auto language detection) to build from source.
- **GPU Configuration**: Updated `docker-compose.yml` to expose all GPUs (`count: all`) to the container by default, enabling the new autodetection feature.

### Added
- **Smart GPU Autodetection**: Implemented logic in `api/config.py` to automatically scan all available CUDA devices and select the one with the most free memory. This prevents startup failures on multi-GPU systems where the default device (GPU 0) might be fully utilized.
