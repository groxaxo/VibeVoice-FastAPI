# Credits

## VibeVoice Standalone - Acknowledgments

This standalone implementation builds upon the excellent work of multiple contributors in the open-source community.

### Original Model & Research

**Microsoft Research - VibeVoice Team**
- **Project**: [VibeVoice](https://github.com/microsoft/VibeVoice)
- **Contribution**: Original VibeVoice TTS model and research
- **License**: MIT License (model subject to Microsoft Research License)
- **Paper**: VibeVoice: Expressive Zero-Shot Text-to-Speech with Multi-Speaker Control

The VibeVoice model represents groundbreaking work in zero-shot text-to-speech synthesis with exceptional voice cloning capabilities.

### ComfyUI Integration

**Fabio Sarracino (FabioSarracino) - VibeVoice-ComfyUI**
- **Project**: [VibeVoice-ComfyUI](https://github.com/Enemyx-net/VibeVoice-ComfyUI)
- **Contribution**: ComfyUI node implementation and integration
- **License**: MIT License

This standalone version is adapted from the VibeVoice-ComfyUI wrapper, which provided:
- Model loading and management logic
- Text chunking for long-form generation
- Voice cloning implementation
- Multi-speaker conversation support
- Pause tag parsing

### Additional Libraries & Frameworks

**Gradio Team**
- **Project**: [Gradio](https://gradio.app)
- **Contribution**: Web interface framework
- **License**: Apache 2.0

**FastAPI & Uvicorn**
- **Project**: [FastAPI](https://fastapi.tiangolo.com)
- **Contribution**: REST API framework
- **License**: MIT License

**Hugging Face Transformers**
- **Project**: [Transformers](https://huggingface.co/transformers)
- **Contribution**: Model loading and inference infrastructure
- **License**: Apache 2.0

**PyTorch Team**
- **Project**: [PyTorch](https://pytorch.org)
- **Contribution**: Deep learning framework
- **License**: BSD-style license

### Standalone Implementation

**This Repository**
- **Adaptation**: Standalone version independent of ComfyUI
- **New Features**:
  - Gradio web interface with all parameters
  - FastAPI REST API with comprehensive endpoints
  - Conda environment support
  - Unified documentation
  - GPU selection via environment variables
  - Text chunking for unlimited length generation

### Community Contributors

Special thanks to:
- The open-source AI/ML community
- Contributors to the original VibeVoice-ComfyUI project
- Users who provided feedback and testing

---

## License Information

This standalone implementation is released under the **MIT License**, inheriting from the VibeVoice-ComfyUI wrapper.

**Important Notes**:
- The VibeVoice model itself is subject to Microsoft's research license
- Please review the original [VibeVoice repository](https://github.com/microsoft/VibeVoice) for model usage terms
- Commercial use should comply with Microsoft's model license terms

---

## Citation

If you use this work in research or production, please consider citing:

**VibeVoice (Original Model)**:
```bibtex
@article{vibevoice2024,
  title={VibeVoice: Expressive Zero-Shot Text-to-Speech with Multi-Speaker Control},
  author={Microsoft Research},
  year={2024}
}
```

**VibeVoice-ComfyUI (Integration)**:
```
VibeVoice-ComfyUI by Fabio Sarracino
https://github.com/Enemyx-net/VibeVoice-ComfyUI
```

---

Thank you to everyone who contributed to making this technology accessible and easy to use! 🎉
