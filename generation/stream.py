from orpheus_tts import OrpheusModel
import os
import wave
import time
import click
from dotenv import load_dotenv
from huggingface_hub import login


# fmt: off
@click.command()
@click.option("--model-name", "-m", default="canopylabs/orpheus-3b-0.1-ft", help="The name of the Orpheus finetuned model to use")
@click.option("--voice", "-v", type=click.Choice(["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]), default="tara", help="Voice to use for speech generation")
@click.option("--output", "-o", default="output.wav", help="Output WAV file path")
@click.option("--prompt", "-p" ,help="Text to synthesize")
# fmt: on
def stream_speech(model_name: str, voice: str, output: str, prompt: str):
    """Generate speech from text using Orpheus TTS."""

    # Initialize environment
    load_dotenv()

    # Log into huggingface in case access limited models need to be fetched
    if huggingface_token := os.environ.get("HF_TOKEN"):
        login(huggingface_token)

    click.echo(f"Initializing model: {model_name}")
    model = OrpheusModel(model_name=model_name)

    click.echo(f"Generating speech with voice '{voice}'...")
    start_time = time.monotonic()
    syn_tokens = model.generate_speech(
        prompt=prompt,
        voice=voice,
    )

    # Create directory if it does not exist yet
    outdir = os.path.dirname(output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with wave.open(output, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        chunk_counter = 0
        click.echo("Streaming speech output...")
        for audio_chunk in syn_tokens:  # output streaming
            chunk_counter += 1
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)
        duration = total_frames / wf.getframerate()

        end_time = time.monotonic()
        click.echo(f"Generated {duration:.2f} seconds of audio at {output}")
        click.echo(f"Generation time: {end_time - start_time:.2f} seconds")
        click.echo(f"Real-time factor: {(end_time - start_time) / duration:.2f}x")


if __name__ == "__main__":
    stream_speech()
