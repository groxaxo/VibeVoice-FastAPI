"""
VibeVoice Gradio Frontend (API Client)
This interface connects to the running VibeVoice FastAPI server.
"""

import gradio as gr
import requests
import json
import os
import base64
import time
import numpy as np
from typing import List, Dict, Any, Generator, Optional

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")


class VibeVoiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.voices = []

    def check_health(self) -> bool:
        """Check if the API is reachable."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def get_voices(self) -> List[str]:
        """Fetch available voice names."""
        try:
            response = requests.get(f"{self.base_url}/v1/vibevoice/voices")
            if response.status_code == 200:
                data = response.json()
                self.voices = [v["name"] for v in data.get("voices", [])]
                return sorted(self.voices)
        except Exception as e:
            print(f"Error fetching voices: {e}")
        return []

    def get_voice_details(self) -> Dict:
        """Fetch full voice details."""
        try:
            response = requests.get(f"{self.base_url}/v1/vibevoice/voices")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            return {"error": str(e)}
        return {}

    def file_to_base64(self, file_path: str) -> Optional[str]:
        """Convert audio file to base64."""
        if not file_path:
            return None
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded

    def generate_openai(self, text, voice, speed, response_format, language, model):
        """Generate using OpenAI-compatible endpoint."""
        # Validate voice
        if not voice:
            # Try to fetch voices if list is empty, or use a default
            if not self.voices:
                self.get_voices()

            if self.voices:
                voice = self.voices[0]
                print(f"Warning: No voice selected, defaulting to {voice}")
            else:
                raise gr.Error(
                    "Please select a voice. (No voices found - is the API running?)"
                )

        url = f"{self.base_url}/v1/audio/speech"
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
            "language": language if language != "auto" else None,
        }

        try:
            start_time = time.time()
            response = requests.post(url, json=payload, stream=True)

            if response.status_code != 200:
                raise gr.Error(f"API Error: {response.text}")

            output_path = f"speech_{int(time.time())}.{response_format}"
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return output_path, f"✅ Generated in {time.time() - start_time:.2f}s"
        except Exception as e:
            raise gr.Error(f"Connection Error: {str(e)}")

    def generate_vibevoice(
        self, script, speakers_config, cfg_scale, steps, seed, stream
    ):
        """Generate using VibeVoice advanced endpoint."""
        url = f"{self.base_url}/v1/vibevoice/generate"

        payload = {
            "script": script,
            "speakers": speakers_config,
            "cfg_scale": cfg_scale,
            "inference_steps": int(steps),
            "seed": int(seed) if seed > -1 else None,
            "stream": stream,
            "response_format": "mp3",
        }

        try:
            if stream:
                response = requests.post(url, json=payload, stream=True)
                if response.status_code != 200:
                    raise gr.Error(f"API Error: {response.text}")

                partial_path = f"stream_{int(time.time())}.mp3"
                chunk_count = 0

                # Handle SSE or Chunked
                is_sse = "text/event-stream" in response.headers.get("content-type", "")

                with open(partial_path, "wb") as f:
                    start_time = time.time()

                    if is_sse:
                        import sseclient

                        client = sseclient.SSEClient(response)
                        for event in client.events():
                            if event.event == "audio":
                                chunk_data = base64.b64decode(event.data)
                                f.write(chunk_data)
                                f.flush()
                                chunk_count += 1
                                if chunk_count % 3 == 0:
                                    yield (
                                        partial_path,
                                        f"Streaming... ({chunk_count} chunks)",
                                    )
                            elif event.event == "error":
                                raise gr.Error(f"Stream Error: {event.data}")
                    else:
                        for chunk in response.iter_content(chunk_size=4096):
                            if chunk:
                                f.write(chunk)
                                f.flush()
                                chunk_count += 1
                                if chunk_count % 10 == 0:
                                    yield (
                                        partial_path,
                                        f"Streaming... ({chunk_count} chunks)",
                                    )

                yield (
                    partial_path,
                    f"✅ Streaming Complete in {time.time() - start_time:.2f}s",
                )
            else:
                start_time = time.time()
                response = requests.post(url, json=payload)
                if response.status_code != 200:
                    raise gr.Error(f"API Error: {response.text}")

                output_path = f"vibevoice_{int(time.time())}.mp3"
                with open(output_path, "wb") as f:
                    f.write(response.content)

                duration = response.headers.get("X-Audio-Duration", "?")
                yield (
                    output_path,
                    f"✅ Generated in {time.time() - start_time:.2f}s (Duration: {duration}s)",
                )

        except Exception as e:
            raise gr.Error(f"Connection Error: {str(e)}")


# Initialize Client
client = VibeVoiceClient(API_BASE_URL)

# --- UI Construction ---
custom_css = """
.gradio-container { font-family: 'Inter', sans-serif; }
.header { 
    text-align: center; 
    padding: 2rem; 
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); 
    color: white; 
    border-radius: 12px; 
    margin-bottom: 2rem;
}
.header h1 { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; }
.header p { opacity: 0.9; font-size: 1.1rem; }
.status-online { color: #22c55e; font-weight: bold; }
.status-offline { color: #ef4444; font-weight: bold; }
"""

with gr.Blocks(title="VibeVoice Studio", css=custom_css, theme=gr.themes.Soft()) as app:
    gr.HTML("""
        <div class="header">
            <h1>VibeVoice Studio</h1>
            <p>Advanced Multi-Speaker TTS & Voice Cloning Interface</p>
        </div>
    """)

    # Status Bar
    with gr.Row():
        api_status = gr.Markdown("Connecting to API...")
        refresh_status_btn = gr.Button("🔄 Check Connection", size="sm")

    # State
    voice_list = gr.State([])

    def update_system_status():
        is_up = client.check_health()
        voices = client.get_voices() if is_up else []
        status_text = (
            f"✅ **API Online** ({len(voices)} voices loaded)"
            if is_up
            else "❌ **API Offline** - Please start the server (./start.sh)"
        )
        return (
            status_text,
            voices,
            gr.update(choices=voices),
            gr.update(choices=voices),
            gr.update(choices=voices),
            gr.update(choices=voices),
            gr.update(choices=voices),
        )

    with gr.Tabs():
        # --- TAB 1: STORYTELLER (Advanced) ---
        with gr.TabItem("🎙️ Storyteller (Multi-Speaker)"):
            with gr.Row():
                with gr.Column(scale=4):
                    script_input = gr.Textbox(
                        label="Dialogue Script",
                        placeholder="Speaker 0: Hello, how are you today?\nSpeaker 1: I'm doing great, thanks for asking!\nSpeaker 0: That is wonderful news.",
                        lines=15,
                        elem_id="script-box",
                    )

                    with gr.Row():
                        num_speakers = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=2,
                            step=1,
                            label="Active Speakers",
                        )
                        use_streaming = gr.Checkbox(
                            label="Real-time Streaming", value=False
                        )
                        generate_btn = gr.Button(
                            "✨ Generate Audio", variant="primary", size="lg"
                        )

                with gr.Column(scale=3):
                    # Speaker Config
                    for i in range(4):
                        with gr.Group(visible=True if i < 2 else False) as spk_group:
                            gr.Markdown(f"**Speaker {i} Configuration**")
                            with gr.Row():
                                preset = gr.Dropdown(
                                    label="Voice Preset",
                                    choices=[],
                                    value=None,
                                    scale=2,
                                )
                                clone_file = gr.File(
                                    label="Clone (Upload Audio)",
                                    type="filepath",
                                    file_types=["audio"],
                                    scale=1,
                                )

                            # Store references to update visibility later
                            if i == 0:
                                spk0_group = spk_group
                                spk0_preset = preset
                                spk0_file = clone_file
                            if i == 1:
                                spk1_group = spk_group
                                spk1_preset = preset
                                spk1_file = clone_file
                            if i == 2:
                                spk2_group = spk_group
                                spk2_preset = preset
                                spk2_file = clone_file
                            if i == 3:
                                spk3_group = spk_group
                                spk3_preset = preset
                                spk3_file = clone_file

                    # Ensure variables are defined even if loop didn't run (unlikely)
                    if "spk0_group" not in locals():
                        spk0_group, spk0_preset, spk0_file = None, None, None
                    if "spk1_group" not in locals():
                        spk1_group, spk1_preset, spk1_file = None, None, None
                    if "spk2_group" not in locals():
                        spk2_group, spk2_preset, spk2_file = None, None, None
                    if "spk3_group" not in locals():
                        spk3_group, spk3_preset, spk3_file = None, None, None

                    with gr.Accordion("Advanced Settings", open=False):
                        cfg_scale = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.3,
                            label="CFG Scale (Adherence)",
                            step=0.1,
                        )
                        steps = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=10,
                            step=1,
                            label="Inference Steps",
                        )
                        seed = gr.Number(value=-1, label="Seed (-1 for random)")

            # Output Area
            with gr.Row():
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
                logs = gr.Markdown("Ready to generate.")

            # Logic
            def update_speaker_visibility(count):
                return {
                    spk2_group: gr.update(visible=count >= 3),
                    spk3_group: gr.update(visible=count >= 4),
                }

            num_speakers.change(
                update_speaker_visibility,
                inputs=num_speakers,
                outputs=[spk2_group, spk3_group],
            )

            def run_vibevoice(
                script,
                n_spk,
                s0p,
                s0f,
                s1p,
                s1f,
                s2p,
                s2f,
                s3p,
                s3f,
                cfg,
                st,
                sd,
                stream,
            ):
                speakers = []

                def add_spk(idx, p, f):
                    if idx >= n_spk:
                        return
                    s_conf = {"speaker_id": idx}
                    if f:
                        s_conf["voice_sample_base64"] = client.file_to_base64(f)
                    elif p:
                        s_conf["voice_preset"] = p
                    else:
                        raise gr.Error(f"Speaker {idx} needs a voice preset or file.")
                    speakers.append(s_conf)

                add_spk(0, s0p, s0f)
                add_spk(1, s1p, s1f)
                add_spk(2, s2p, s2f)
                add_spk(3, s3p, s3f)

                yield from client.generate_vibevoice(
                    script, speakers, cfg, st, sd, stream
                )

            generate_btn.click(
                run_vibevoice,
                inputs=[
                    script_input,
                    num_speakers,
                    spk0_preset,
                    spk0_file,
                    spk1_preset,
                    spk1_file,
                    spk2_preset,
                    spk2_file,
                    spk3_preset,
                    spk3_file,
                    cfg_scale,
                    steps,
                    seed,
                    use_streaming,
                ],
                outputs=[audio_output, logs],
            )

        # --- TAB 2: SIMPLE TTS (OpenAI) ---
        with gr.TabItem("💬 Simple TTS (OpenAI Compatible)"):
            with gr.Row():
                with gr.Column():
                    oa_input = gr.Textbox(label="Text", lines=5, value="Hello world.")
                    oa_voice = gr.Dropdown(
                        label="Voice", choices=[], allow_custom_value=True
                    )
                    with gr.Row():
                        oa_speed = gr.Slider(0.25, 4.0, 1.0, step=0.1, label="Speed")
                        oa_lang = gr.Dropdown(
                            ["auto", "en", "es", "fr", "de", "zh", "ja"],
                            value="auto",
                            label="Language",
                        )
                        oa_model = gr.Dropdown(
                            ["tts-1", "tts-1-hd"], value="tts-1", label="Model"
                        )
                    oa_btn = gr.Button("Generate", variant="primary")

                with gr.Column():
                    oa_audio = gr.Audio(type="filepath")
                    oa_status = gr.Markdown("")

            oa_btn.click(
                client.generate_openai,
                inputs=[
                    oa_input,
                    oa_voice,
                    oa_speed,
                    gr.State("mp3"),
                    oa_lang,
                    oa_model,
                ],
                outputs=[oa_audio, oa_status],
            )

        # --- TAB 3: GALLERY ---
        with gr.TabItem("📚 Voice Library"):
            refresh_lib_btn = gr.Button("Refresh Library")
            lib_view = gr.JSON(label="Voice Details")
            refresh_lib_btn.click(client.get_voice_details, outputs=lib_view)

    # Init
    app.load(
        update_system_status,
        outputs=[
            api_status,
            voice_list,
            oa_voice,
            spk0_preset,
            spk1_preset,
            spk2_preset,
            spk3_preset,
        ],
    )
    refresh_status_btn.click(
        update_system_status,
        outputs=[
            api_status,
            voice_list,
            oa_voice,
            spk0_preset,
            spk1_preset,
            spk2_preset,
            spk3_preset,
        ],
    )

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", 7860))
    app.queue().launch(
        server_name="0.0.0.0", server_port=port, share=False, show_error=True
    )
