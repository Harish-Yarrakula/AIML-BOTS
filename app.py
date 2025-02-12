
import gradio as gr
import assemblyai as aai
from together import Together
import base64
from io import BytesIO
from PIL import Image
import os
import yaml


# Function to load API credentials
def load_credentials():
    assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")
    
    if not assemblyai_key or not together_key:
        try:
            with open('API.yml', 'r') as file:
                api_creds = yaml.safe_load(file)
                assemblyai_key = assemblyai_key or api_creds['assemblyai']
                together_key = together_key or api_creds['Together_api']
        except Exception as e:
            print(f"Failed to load API credentials: {str(e)}")
            return None, None
    
    return assemblyai_key, together_key

# Initialize API clients
ASSEMBLYAI_API_KEY, TOGETHER_API_KEY = load_credentials()

if ASSEMBLYAI_API_KEY and TOGETHER_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY
    together_client = Together(api_key=TOGETHER_API_KEY)
else:
    raise ValueError("API credentials not found. Please check your configuration.")

def transcribe_audio(audio_path):
    """Transcribe audio using AssemblyAI."""
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path)
        return transcript.text
    except Exception as e:
        return f"Error in transcription: {str(e)}"

def generate_image(prompt):
    """Generate image using Together AI."""
    try:
        response = together_client.images.generate(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell-Free",
            width=1024,
            height=768,
            steps=4,
            n=1,
            response_format="b64_json"
        )
        # Convert base64 to PIL Image
        img_data = base64.b64decode(response.data[0].b64_json)
        img = Image.open(BytesIO(img_data))
        return img
    except Exception as e:
        return f"Error in image generation: {str(e)}"

def process_audio(audio, progress=gr.Progress()):
    """Process audio file and generate image"""
    if audio is None:
        return None, "Please provide an audio input."
    
    progress(0.3, desc="Transcribing audio...")
    transcribed_text = transcribe_audio(audio)
    
    if isinstance(transcribed_text, str) and not transcribed_text.startswith("Error"):
        progress(0.6, desc="Generating image...")
        generated_image = generate_image(transcribed_text)
        
        if isinstance(generated_image, Image.Image):
            progress(1.0, desc="Complete!")
            return generated_image, transcribed_text
        else:
            return None, f"Image generation failed: {generated_image}"
    else:
        return None, f"Transcription failed: {transcribed_text}"

# Custom CSS for better styling
custom_css = """
#app-title {
    text-align: center;
    margin-bottom: 10px;
}
#app-subtitle {
    text-align: center;
    margin-bottom: 30px;
}
#main-container {
    max-width: 1200px;
    margin: auto;
}
"""

# Create Gradio interface
def create_interface():
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate"
    )) as app:
        gr.HTML(
            """
            <div id="app-title">
                <h1>AIML-BOTS</h1>
                <h1> Audio2Img </h1>
            </div>
            <div id="app-subtitle">
                <h3>✨ Transform Your Words into Stunning Visual Art ✨</h3>
            </div>
            """
        )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Record or Upload Audio",
                    sources=["microphone", "upload"],
                    type="filepath"
                )
                submit_btn = gr.Button("🚀 Generate Vision", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Generated Image 🖼️")
                output_text = gr.Textbox(
                    label="Transcribed Text 📝",
                    placeholder="Your speech will appear here...",
                    lines=3
                )

        # Add usage instructions

        submit_btn.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[output_image, output_text]
        )

    return app

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)