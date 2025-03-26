import json
import numpy as np
import librosa
from datasets import Dataset
import os
import click


# fmt: off
@click.command()
@click.option('--audio', '-a', type=click.Path(exists=True, dir_okay=False), required=True, help='Path to the audio file')
@click.option('--transcription', '-t', type=click.Path(exists=True, dir_okay=False), required=True, help='Path to the transcription JSON file')
@click.option('--output', '-o', type=click.Path(exists=True), default='segmented_data.parquet', help='Output parquet file path')
@click.option('--sample-rate', '-sr', default=24000, type=int, help='Target sample rate in Hz')
@click.option('--max-length', '-ml', default=14.0, type=float, help='Max audio snippet length (snippets that are longer will be discarded)')
# fmt: on
def process_audio_with_transcription(audio, transcription, output, sample_rate):
    """Process audio file with JSON transcription and save segments as parquet dataset."""

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
    with click.progressbar(
        transcription_data["segments"], label="Processing segments"
    ) as segments:
        # Process each segment
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
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
    dataset.save_to_disk(output)

    click.echo(
        click.style(
            f"✓ Successfully saved {segments_count} segments to {output}", fg="green"
        )
    )
    return dataset


if __name__ == "__main__":
    process_audio_with_transcription()
