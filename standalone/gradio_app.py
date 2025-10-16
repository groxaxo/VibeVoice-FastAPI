"""
Gradio Interface for VibeVoice TTS
Standalone version with all features from ComfyUI nodes
"""

import gradio as gr
import os
import tempfile
import soundfile as sf
import logging
from pathlib import Path
from vibevoice_core import VibeVoiceCore, get_optimal_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VibeVoiceApp")

# Initialize VibeVoice Core
models_dir = os.environ.get("VIBEVOICE_MODELS_DIR", "./models")
core = VibeVoiceCore(models_dir=models_dir)

# Get available models on startup
available_models = core.get_available_models()
if not available_models:
    logger.warning("No models found. Please place VibeVoice models in the models directory.")
    available_models = ["No models found"]

# Set default model (prefer VibeVoice-Large if available)
default_model = available_models[0]
if "VibeVoice-Large" in available_models:
    default_model = "VibeVoice-Large"

# Get device info
device = get_optimal_device()
logger.info(f"Using device: {device}")

def generate_single_speaker(
    text, model_name, voice_file, attention_type, quantize_llm,
    free_memory, diffusion_steps, seed, cfg_scale, use_sampling,
    temperature, top_p, voice_speed_factor
):
    """Generate single speaker speech"""
    try:
        if not text or not text.strip():
            return None, "Error: No text provided"
        
        if model_name == "No models found":
            return None, "Error: No models available. Please download models first."
        
        # Load model
        core.load_model(
            model_name=model_name,
            attention_type=attention_type,
            quantize_llm=quantize_llm
        )
        
        # Prepare voice sample
        voice_samples = None
        if voice_file is not None:
            voice_sample = core._prepare_audio_from_file(
                voice_file, 
                speed_factor=voice_speed_factor
            )
            if voice_sample is not None:
                voice_samples = [voice_sample]
        
        # Generate speech
        audio_array, sample_rate = core.generate_speech(
            text=text,
            voice_samples=voice_samples,
            cfg_scale=cfg_scale,
            seed=seed,
            diffusion_steps=diffusion_steps,
            use_sampling=use_sampling,
            temperature=temperature,
            top_p=top_p,
            num_speakers=1
        )
        
        # Free memory if requested
        if free_memory:
            core.free_memory()
        
        # Save to temporary file for Gradio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, audio_array, sample_rate)
            output_path = f.name
        
        return output_path, f"✅ Generated successfully! ({len(text.split())} words)"
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return None, f"❌ Error: {str(e)}"

def generate_multi_speaker(
    text, model_name, speaker1_voice, speaker2_voice, speaker3_voice, speaker4_voice,
    attention_type, quantize_llm, free_memory, diffusion_steps, seed, cfg_scale,
    use_sampling, temperature, top_p, voice_speed_factor
):
    """Generate multi-speaker conversation"""
    try:
        if not text or not text.strip():
            return None, "Error: No text provided"
        
        if model_name == "No models found":
            return None, "Error: No models available. Please download models first."
        
        # Detect number of speakers from text
        import re
        bracket_pattern = r'\[(\d+)\]\s*:'
        speakers_numbers = sorted(list(set([int(m) for m in re.findall(bracket_pattern, text)])))
        
        if not speakers_numbers:
            return None, "Error: No speaker labels found. Use format: [1]: Text here [2]: More text"
        
        num_speakers = max(speakers_numbers)
        if num_speakers > 4:
            return None, "Error: Maximum 4 speakers supported"
        
        # Load model
        core.load_model(
            model_name=model_name,
            attention_type=attention_type,
            quantize_llm=quantize_llm
        )
        
        # Prepare voice samples for each speaker
        voice_files = [speaker1_voice, speaker2_voice, speaker3_voice, speaker4_voice]
        voice_samples = []
        
        for i, speaker_num in enumerate(speakers_numbers):
            idx = speaker_num - 1
            
            if idx < len(voice_files) and voice_files[idx] is not None:
                voice_sample = core._prepare_audio_from_file(
                    voice_files[idx],
                    speed_factor=voice_speed_factor
                )
                if voice_sample is None:
                    voice_sample = core._create_synthetic_voice_sample(idx)
            else:
                voice_sample = core._create_synthetic_voice_sample(idx)
            
            voice_samples.append(voice_sample)
        
        # Convert [N]: format to Speaker N: format for processing
        converted_text = text
        for speaker_num in sorted(speakers_numbers, reverse=True):
            pattern = f'\\[{speaker_num}\\]\\s*:'
            replacement = f'Speaker {speaker_num}:'
            converted_text = re.sub(pattern, replacement, converted_text)
        
        # Generate speech
        audio_array, sample_rate = core.generate_speech(
            text=converted_text,
            voice_samples=voice_samples,
            cfg_scale=cfg_scale,
            seed=seed,
            diffusion_steps=diffusion_steps,
            use_sampling=use_sampling,
            temperature=temperature,
            top_p=top_p,
            num_speakers=num_speakers
        )
        
        # Free memory if requested
        if free_memory:
            core.free_memory()
        
        # Save to temporary file for Gradio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, audio_array, sample_rate)
            output_path = f.name
        
        return output_path, f"✅ Generated successfully! ({num_speakers} speakers, {len(text.split())} words)"
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return None, f"❌ Error: {str(e)}"

# Build Gradio Interface
with gr.Blocks(title="VibeVoice TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ VibeVoice Text-to-Speech")
    gr.Markdown(f"**Device:** {device} | **Models Directory:** `{models_dir}`")
    
    with gr.Tabs():
        # Single Speaker Tab
        with gr.Tab("🎤 Single Speaker"):
            with gr.Row():
                with gr.Column(scale=2):
                    ss_text = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter text here... You can use [pause] or [pause:1500] for silences.",
                        lines=5,
                        value="Hello, this is a test of the VibeVoice text-to-speech system."
                    )
                    
                    with gr.Row():
                        ss_model = gr.Dropdown(
                            label="Model",
                            choices=available_models,
                            value=default_model
                        )
                        ss_attention = gr.Dropdown(
                            label="Attention Type",
                            choices=["auto", "eager", "sdpa", "flash_attention_2"],
                            value="flash_attention_2"
                        )
                    
                    with gr.Row():
                        ss_quantize = gr.Dropdown(
                            label="LLM Quantization",
                            choices=["full precision", "4bit", "8bit"],
                            value="full precision"
                        )
                        ss_diffusion_steps = gr.Slider(
                            label="Diffusion Steps",
                            minimum=1,
                            maximum=100,
                            value=20,
                            step=1
                        )
                    
                    with gr.Row():
                        ss_seed = gr.Number(label="Seed", value=42, precision=0)
                        ss_cfg_scale = gr.Slider(
                            label="CFG Scale",
                            minimum=0.5,
                            maximum=3.5,
                            value=1.3,
                            step=0.05
                        )
                    
                    with gr.Accordion("Advanced Options", open=False):
                        ss_use_sampling = gr.Checkbox(label="Use Sampling", value=False)
                        ss_temperature = gr.Slider(
                            label="Temperature (sampling only)",
                            minimum=0.1,
                            maximum=2.0,
                            value=0.95,
                            step=0.05
                        )
                        ss_top_p = gr.Slider(
                            label="Top P (sampling only)",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.95,
                            step=0.05
                        )
                        ss_voice_speed = gr.Slider(
                            label="Voice Speed Factor",
                            minimum=0.8,
                            maximum=1.2,
                            value=1.0,
                            step=0.01
                        )
                        ss_free_memory = gr.Checkbox(
                            label="Free Memory After Generation",
                            value=True
                        )
                    
                    ss_voice_file = gr.Audio(
                        label="Voice to Clone (Optional)",
                        type="filepath"
                    )
                    
                    ss_generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    ss_audio_output = gr.Audio(label="Generated Audio", type="filepath")
                    ss_status = gr.Textbox(label="Status", lines=2)
            
            ss_generate_btn.click(
                fn=generate_single_speaker,
                inputs=[
                    ss_text, ss_model, ss_voice_file, ss_attention, ss_quantize,
                    ss_free_memory, ss_diffusion_steps, ss_seed, ss_cfg_scale,
                    ss_use_sampling, ss_temperature, ss_top_p, ss_voice_speed
                ],
                outputs=[ss_audio_output, ss_status]
            )
        
        # Multi Speaker Tab
        with gr.Tab("👥 Multi-Speaker"):
            with gr.Row():
                with gr.Column(scale=2):
                    ms_text = gr.Textbox(
                        label="Multi-Speaker Conversation",
                        placeholder="Use [1]:, [2]:, [3]:, [4]: for speaker labels\nExample:\n[1]: Hello, how are you?\n[2]: I'm great, thanks!",
                        lines=8,
                        value="[1]: Hello, this is the first speaker.\n[2]: Hi there, I'm the second speaker.\n[1]: Nice to meet you!\n[2]: Nice to meet you too!"
                    )
                    
                    with gr.Row():
                        ms_model = gr.Dropdown(
                            label="Model",
                            choices=available_models,
                            value=default_model
                        )
                        ms_attention = gr.Dropdown(
                            label="Attention Type",
                            choices=["auto", "eager", "sdpa", "flash_attention_2"],
                            value="flash_attention_2"
                        )
                    
                    with gr.Row():
                        ms_quantize = gr.Dropdown(
                            label="LLM Quantization",
                            choices=["full precision", "4bit", "8bit"],
                            value="full precision"
                        )
                        ms_diffusion_steps = gr.Slider(
                            label="Diffusion Steps",
                            minimum=1,
                            maximum=100,
                            value=20,
                            step=1
                        )
                    
                    with gr.Row():
                        ms_seed = gr.Number(label="Seed", value=42, precision=0)
                        ms_cfg_scale = gr.Slider(
                            label="CFG Scale",
                            minimum=0.5,
                            maximum=3.5,
                            value=1.3,
                            step=0.05
                        )
                    
                    with gr.Accordion("Advanced Options", open=False):
                        ms_use_sampling = gr.Checkbox(label="Use Sampling", value=False)
                        ms_temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.1,
                            maximum=2.0,
                            value=0.95,
                            step=0.05
                        )
                        ms_top_p = gr.Slider(
                            label="Top P",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.95,
                            step=0.05
                        )
                        ms_voice_speed = gr.Slider(
                            label="Voice Speed Factor",
                            minimum=0.8,
                            maximum=1.2,
                            value=1.0,
                            step=0.01
                        )
                        ms_free_memory = gr.Checkbox(
                            label="Free Memory After Generation",
                            value=True
                        )
                    
                    with gr.Row():
                        ms_speaker1_voice = gr.Audio(label="Speaker 1 Voice", type="filepath")
                        ms_speaker2_voice = gr.Audio(label="Speaker 2 Voice", type="filepath")
                    
                    with gr.Row():
                        ms_speaker3_voice = gr.Audio(label="Speaker 3 Voice", type="filepath")
                        ms_speaker4_voice = gr.Audio(label="Speaker 4 Voice", type="filepath")
                    
                    ms_generate_btn = gr.Button("🎵 Generate Conversation", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    ms_audio_output = gr.Audio(label="Generated Audio", type="filepath")
                    ms_status = gr.Textbox(label="Status", lines=3)
            
            ms_generate_btn.click(
                fn=generate_multi_speaker,
                inputs=[
                    ms_text, ms_model, ms_speaker1_voice, ms_speaker2_voice,
                    ms_speaker3_voice, ms_speaker4_voice, ms_attention, ms_quantize,
                    ms_free_memory, ms_diffusion_steps, ms_seed, ms_cfg_scale,
                    ms_use_sampling, ms_temperature, ms_top_p, ms_voice_speed
                ],
                outputs=[ms_audio_output, ms_status]
            )
        
        # Info Tab
        with gr.Tab("ℹ️ Information"):
            gr.Markdown("""
            ## VibeVoice Standalone TTS
            
            ### Features
            - **Single Speaker TTS**: Generate natural speech with optional voice cloning
            - **Multi-Speaker Conversations**: Support for up to 4 distinct speakers
            - **Voice Cloning**: Clone voices from audio samples
            - **Custom Pause Tags**: Use `[pause]` or `[pause:1500]` to insert silences
            - **Flexible Configuration**: Control temperature, sampling, and guidance scale
            
            ### Setup
            1. **Install Dependencies**: Use the provided `environment.yml` or `requirements.txt`
            2. **Download Models**: Place VibeVoice models in the `models/` directory
            3. **Download Tokenizer**: Place Qwen2.5-1.5B tokenizer files in `models/tokenizer/`
            
            ### Model Directory Structure
            ```
            models/
            ├── tokenizer/                     # Qwen tokenizer (required)
            │   ├── tokenizer_config.json
            │   ├── vocab.json
            │   ├── merges.txt
            │   └── tokenizer.json
            ├── VibeVoice-1.5B/               # Your models
            │   ├── config.json
            │   ├── model-*.safetensors
            │   └── ...
            └── VibeVoice-Large/
                └── ...
            ```
            
            ### Multi-Speaker Format
            Use `[N]:` notation where N is 1-4:
            ```
            [1]: Hello, how are you today?
            [2]: I'm doing great, thanks for asking!
            [1]: That's wonderful to hear.
            ```
            
            ### Tips
            - **Seed Management**: Save good seeds for consistent voices
            - **Model Selection**: Use Large models for better multi-speaker quality
            - **Performance**: First run loads the model (may take time)
            - **Memory**: Enable "Free Memory After Generation" to save VRAM
            
            ### Device Information
            - **Current Device**: {device}
            - **Models Directory**: {models_dir}
            - **Available Models**: {model_count}
            """.format(device=device, models_dir=models_dir, model_count=len(available_models)))
    
    gr.Markdown("---")
    gr.Markdown("Built with ❤️ using [VibeVoice](https://github.com/microsoft/VibeVoice) and [Gradio](https://gradio.app)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VibeVoice Gradio Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--models-dir", default="./models", help="Models directory path")
    
    args = parser.parse_args()
    
    if args.models_dir:
        os.environ["VIBEVOICE_MODELS_DIR"] = args.models_dir
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
