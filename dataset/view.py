from typing import Optional
import click
from datasets import load_dataset
from renumics import spotlight
import os
from pathlib import Path


# fmt: off
@click.command()
@click.option("--format", "-f", type=click.Choice(["csv", "json", "parquet"]), help="Specify the file format explicitly. If not provided, will be inferred from file extension.")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
# fmt: on
def open_dataset(format: Optional[str], file_path: Path):
    """Open a dataset from a CSV or Parquet file and display it with Spotlight."""

    # Infer format from extension if not provided
    if not format:
        if file_path.suffix.lower() == ".csv":
            format = "csv"
        elif file_path.suffix.lower() == ".json":
            format = "json"
        elif file_path.suffix.lower() in [".parquet", ".pq"]:
            format = "parquet"
        else:
            raise click.BadParameter(
                f"Could not infer format from file extension '{file_path.suffix}'. "
                "Please specify format using the --format option."
            )

    click.echo(f"Loading {format} dataset from: {file_path}")

    # Load dataset based on format
    dataset = load_dataset(format, data_files=str(file_path))

    # Display dataset with spotlight
    click.echo(
        f"Opening dataset in Spotlight. Dataset has {len(dataset['train'])} samples."
    )
    spotlight.show(dataset["train"])


if __name__ == "__main__":
    open_dataset()
