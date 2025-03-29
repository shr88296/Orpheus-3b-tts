from typing import Optional
import json
import librosa
from datasets import Dataset, Audio
import click
from tqdm import tqdm


# fmt: off
@click.command()
@click.option("--audio", "-a", type=click.Path(exists=True, dir_okay=False), required=True, help="Path to the audio file")
@click.option("--transcription", "-t", type=click.Path(exists=True, dir_okay=False), required=True, help="Path to the transcription JSON file")
@click.option("--output", "-o", type=click.Path(dir_okay=False), default="segmented_data.parquet", help="Output parquet file path")
@click.option("--sample-rate", "-sr", type=int, help="Target sample rate in Hz (orpheus needs 24000)")
@click.option("--min-length", "-ml", type=float, default=0.0, help="Min audio snippet length (snippets that are shorter will be discarded)")
@click.option("--max-length", "-lm", type=float, help="Max audio snippet length (snippets that are longer will be discarded)")
@click.option("--pad-start", "-ps", type=float, default=0.0,  help="How many seconds to add at the start of the snippet")
@click.option("--pad-end", "-pe", type=float, default=0.0,  help="How many seconds to add at the end of the snippet")
# fmt: on
def process_audio_with_transcription(
    audio: str,
    transcription: str,
    output: str,
    sample_rate: Optional[int],
    min_length: float,
    max_length: Optional[float],
    pad_start: float,
    pad_end: float,
):
    """Process audio file with JSON transcription from whisperx and save segments as parquet dataset."""

    click.echo(
        f"Loading audio at {f'{sample_rate}Hz' if sample_rate else 'original'} sample rate..."
    )
    # Load audio file directly at the target sample rate
    audio_data, sample_rate = librosa.load(
        audio, sr=sample_rate, mono=True  # Target sample rate  # Ensure mono audio
    )
    audio_sample_length = len(audio_data)

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

            # Only process audio if it's between min length and max length
            if not max_length or min_length <= end_time - start_time <= max_length:
                text = segment["text"]

                # Convert time to samples and optionally add padding
                start_sample = int(start_time * sample_rate) - int(
                    pad_start * sample_rate
                )
                end_sample = int(end_time * sample_rate) + int(pad_end * sample_rate)
                if start_sample < 0:
                    start_sample = 0
                if end_sample > audio_sample_length - 1:
                    end_sample = audio_sample_length - 1

                # Extract segment audio
                segment_audio = audio_data[start_sample:end_sample]

                # Create dataset item
                dataset_items.append(
                    {
                        "text": text,
                        "audio": {
                            "array": segment_audio,
                            "sampling_rate": sample_rate,
                        },
                    }
                )

    # Create dataset and save as parquet
    dataset = Dataset.from_list(dataset_items).cast_column("audio", Audio())
    dataset.to_parquet(output)

    click.echo(
        click.style(
            f"Successfully saved {len(dataset_items)}/{segments_count} segments to {output}",
            fg="green",
        )
    )


if __name__ == "__main__":
    process_audio_with_transcription()
