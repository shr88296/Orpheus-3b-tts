from typing import Optional
from snac import SNAC
import torch
import numpy as np
import soundfile as sf
import os
import click
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login, snapshot_download


# fmt: off
@click.command()
@click.option("--model-name", "-m", type=str, default="canopylabs/orpheus-3b-0.1-ft", help="Model name/path on HuggingFace")
@click.option("--voice", "-v", type=str, help="Voice to use (see github for available voices)")
@click.option("--output", "-o", type=click.Path(dir_okay=False), default="output.wav", help="Output WAV file path (will be modified with number)")
@click.argument("prompts", nargs=-1, required=True)
# fmt: on
def generate_speech(
    model_name: str, voice: Optional[str], output: str, prompts: list[str]
) -> None:
    """Generate speech from text prompts using Orpheus TTS.

    PROMPTS: One or more text prompts to convert to speech.
    """
    # Initialize environment and login to huggingface
    load_dotenv()

    if huggingface_token := os.environ.get("HF_TOKEN"):
        login(huggingface_token)

    # Ensure output directory exists
    os.makedirs("out", exist_ok=True)

    # Download only model config and safetensors
    model_path = snapshot_download(
        repo_id=model_name,
        allow_patterns=[
            "config.json",
            "*.safetensors",
            "model.safetensors.index.json",
        ],
        ignore_patterns=[
            "optimizer.pt",
            "pytorch_model.bin",
            "training_args.bin",
            "scheduler.pt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.*",
        ],
    )

    # Load SNAC model for audio decoding
    click.echo("Loading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cpu")

    # Load model and tokenizer
    click.echo(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Format prompts into correct template, if voice is given
    if voice:
        click.echo(f"Preparing prompts for voice: {voice}")
        prompts = [f"{voice}: " + p for p in prompts]

    all_input_ids = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        all_input_ids.append(input_ids)

    start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
    end_tokens = torch.tensor(
        [[128009, 128260]], dtype=torch.int64
    )  # End of text, End of human

    all_modified_input_ids = []
    for input_ids in all_input_ids:
        modified_input_ids = torch.cat(
            [start_token, input_ids, end_tokens], dim=1
        )  # SOH SOT Text EOT EOH
        all_modified_input_ids.append(modified_input_ids)

    all_padded_tensors = []
    all_attention_masks = []
    max_length = max(
        [modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids]
    )
    for modified_input_ids in all_modified_input_ids:
        padding = max_length - modified_input_ids.shape[1]
        padded_tensor = torch.cat(
            [torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                torch.zeros((1, padding), dtype=torch.int64),
                torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64),
            ],
            dim=1,
        )
        all_padded_tensors.append(padded_tensor)
        all_attention_masks.append(attention_mask)

    all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)

    input_ids = all_padded_tensors.to("cuda")
    attention_mask = all_attention_masks.to("cuda")

    # Generate Output
    click.echo("Generating speech...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=128258,
        )

    # Parse Output as speech
    token_to_find = 128257
    token_to_remove = 128258

    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx + 1 :]
    else:
        cropped_tensor = generated_ids

    mask = cropped_tensor != token_to_remove

    processed_rows = []

    for row in cropped_tensor:
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)

    code_lists = []

    for row in processed_rows:
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]
        trimmed_row = [t - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)

    def redistribute_codes(code_list):
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list) + 1) // 7):
            layer_1.append(code_list[7 * i])
            layer_2.append(code_list[7 * i + 1] - 4096)
            layer_3.append(code_list[7 * i + 2] - (2 * 4096))
            layer_3.append(code_list[7 * i + 3] - (3 * 4096))
            layer_2.append(code_list[7 * i + 4] - (4 * 4096))
            layer_3.append(code_list[7 * i + 5] - (5 * 4096))
            layer_3.append(code_list[7 * i + 6] - (6 * 4096))
        codes = [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0),
        ]
        audio_hat = snac_model.decode(codes)
        return audio_hat

    prompts_samples = []
    for code_list in code_lists:
        samples = redistribute_codes(code_list)
        prompts_samples.append(samples)

    # Save speech to wav file
    if len(prompts) != len(prompts_samples):
        raise RuntimeError("Number of prompts and samples do not match")
    else:
        # Create directory if it does not exist yet
        outdir = os.path.dirname(output)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        number_width = len(str(len(prompts_samples)))
        for i, samples in enumerate(prompts_samples):

            # Create output filename
            output_root, output_ext = os.path.splitext(output)
            output_filename = f"{output_root}_{str(i).zfill(number_width)}{output_ext}"

            # Convert to the appropriate format
            samples = samples.detach().squeeze().to("cpu").numpy()

            # Save using soundfile - automatically handles format based on extension
            sf.write(
                output_filename,
                samples,
                24000,  # Sample rate
            )

            click.echo(f"Saved audio sample {i} to {output_filename}")


if __name__ == "__main__":
    generate_speech()
