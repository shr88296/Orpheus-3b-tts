from typing import Optional
import json
import librosa
from datasets import Dataset
import click
from tqdm import tqdm


# fmt: off
@click.command()
@click.option('--audio', '-a', type=click.Path(exists=True, dir_okay=False), required=True, help='Path to the audio file')
@click.option('--transcription', '-t', type=click.Path(exists=True, dir_okay=False), required=True, help='Path to the transcription JSON file')
@click.option('--output', '-o', type=click.Path(dir_okay=False), default='segmented_data.parquet', help='Output parquet file path')
@click.option('--sample-rate', '-sr', default=24000, type=int, help='Target sample rate in Hz')
@click.option('--max-length', '-ml', type=float, help='Max audio snippet length (snippets that are longer will be discarded)')
# fmt: on
def process_audio_with_transcription(
    audio: str,
    transcription: str,
    output: str,
    sample_rate: int,
    max_length: Optional[float],
):
    """Process audio file with JSON transcription from whisperx and save segments as parquet dataset."""

    click.echo(f"Loading audio at {sample_rate}Hz sample rate...")
    # Load audio file directly at the target sample rate
    audio_data, _ = librosa.load(
        audio, sr=sample_rate, mono=True  # Target sample rate  # Ensure mono audio
    )

    # Load JSON transcription
    with open(transcription, "r") as f:
        transcription_data = json.load(f)

    segments_count = len(transcription_data["segments"])
    click.echo(f"Processing {segments_count} segments...")

    # Prepare data for the dataset
    dataset_items = []

    # Create progress bar
    with tqdm(transcription_data["segments"], desc="Processing segments") as segments:
        # Process each segment
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]

            # Only process audio if it doesn't exceed the max length
            if not max_length or end_time - start_time <= max_length:
                text = segment["text"]

                # Convert time to samples
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)

                # Extract segment audio
                segment_audio = audio_data[start_sample:end_sample]

                # Create dataset item
                dataset_items.append(
                    {
                        "text": text,
                        "audio": {"array": segment_audio, "sampling_rate": sample_rate},
                    }
                )

    # Create dataset and save as parquet
    dataset = Dataset.from_list(dataset_items)
    dataset.to_parquet(output)

    click.echo(
        click.style(
            f"Successfully saved {len(dataset_items)}/{segments_count} segments to {output}",
            fg="green",
        )
    )


if __name__ == "__main__":
    process_audio_with_transcription()
