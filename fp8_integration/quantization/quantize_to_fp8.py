"""
FP8 Quantization Script for Orpheus-TTS
Uses llm-compressor to perform offline post-training quantization to FP8 format
"""

import os
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.quantization.fp8 import FP8Modifier
from datasets import load_dataset
import yaml
import json


def create_calibration_dataset(tokenizer, num_samples=512, max_length=2048):
    """
    Create a calibration dataset for FP8 quantization.
    This should be representative of the target use case (TTS prompts).
    """
    # Sample TTS prompts covering various voices, emotions, and speech patterns
    sample_prompts = [
        "tara: Hello, how are you today? I hope you're having a wonderful day!",
        "leo: <laugh> That's hilarious! I can't believe that actually happened.",
        "zoe: <sigh> I'm feeling a bit tired today, but I'll manage.",
        "jess: Could you please explain how this works? I'm very curious to learn more.",
        "mia: <chuckle> Oh, that reminds me of a funny story from last week.",
        "julia: The weather is absolutely beautiful today! Perfect for a walk in the park.",
        "leah: <cough> Excuse me, I think I might be coming down with something.",
        "zac: Let me tell you about the amazing adventure I had yesterday!",
        "tara: Can you repeat that? I didn't quite catch what you said.",
        "leo: I'm so excited about the upcoming event! It's going to be fantastic!",
        # Add phonetically diverse sentences
        "tara: The quick brown fox jumps over the lazy dog near the riverbank.",
        "zoe: Peter Piper picked a peck of pickled peppers from the garden.",
        "jess: She sells seashells by the seashore on sunny summer days.",
        "mia: The thirty-three thieves thought that they thrilled the throne throughout Thursday.",
        # Add numerical and special character handling
        "julia: The meeting is scheduled for 3:45 PM on December 21st, 2024.",
        "leah: Please call me at 555-1234 or email john.doe@example.com.",
        "zac: The temperature today is 72°F with 65% humidity.",
        # Add emotional variations
        "tara: <laugh> <laugh> I can't stop laughing at this joke!",
        "leo: <sigh> <sigh> Sometimes life can be quite challenging.",
        "zoe: <chuckle> Well, that's an interesting way to look at it!",
        # Add longer, more complex sentences
        "jess: In the midst of winter, I found there was, within me, an invincible summer that made me realize the importance of inner strength.",
        "mia: The conference will cover topics including artificial intelligence, machine learning, quantum computing, and biotechnology advances.",
    ]
    
    # Expand the dataset by creating variations
    calibration_data = []
    
    # Add custom tokens for TTS
    for prompt in sample_prompts:
        # Format the prompt as the model expects
        prompt_tokens = tokenizer(prompt, return_tensors="pt")
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        prompt_string = tokenizer.decode(all_input_ids[0])
        
        calibration_data.append(prompt_string)
    
    # If we need more samples, create variations
    while len(calibration_data) < num_samples:
        for base_prompt in sample_prompts:
            if len(calibration_data) >= num_samples:
                break
            # Add variations with different punctuation or slight modifications
            variations = [
                base_prompt + " What do you think?",
                base_prompt + " That's interesting!",
                base_prompt + " Let me know your thoughts.",
                base_prompt.replace(".", "!"),
                base_prompt.replace("?", "."),
            ]
            for var in variations:
                if len(calibration_data) >= num_samples:
                    break
                # Format with TTS tokens
                prompt_tokens = tokenizer(var, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = tokenizer.decode(all_input_ids[0])
                calibration_data.append(prompt_string)
    
    # Truncate to exact number of samples
    calibration_data = calibration_data[:num_samples]
    
    # Create dataset format expected by llm-compressor
    dataset_dict = {
        "text": calibration_data,
        "input_ids": [tokenizer.encode(text, max_length=max_length, truncation=True) for text in calibration_data]
    }
    
    return dataset_dict


def quantize_model_to_fp8(
    model_path: str,
    output_path: str,
    tokenizer_path: str = None,
    num_calibration_samples: int = 512,
    max_seq_length: int = 2048,
    fp8_format: str = "fp8_e4m3",  # or "fp8_e5m2"
    quantize_kv_cache: bool = True,
    preserve_critical_layers: bool = True,
):
    """
    Quantize Orpheus-TTS model to FP8 format using llm-compressor.
    
    Args:
        model_path: Path to the base BF16 model
        output_path: Path to save the quantized model
        tokenizer_path: Path to tokenizer (defaults to model_path)
        num_calibration_samples: Number of calibration samples for quantization
        max_seq_length: Maximum sequence length for calibration
        fp8_format: FP8 format to use ("fp8_e4m3" or "fp8_e5m2")
        quantize_kv_cache: Whether to quantize KV cache
        preserve_critical_layers: Keep lm_head and embeddings in BF16
    """
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Load model in BF16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("Creating calibration dataset...")
    calibration_data = create_calibration_dataset(
        tokenizer, 
        num_samples=num_calibration_samples,
        max_length=max_seq_length
    )
    
    # Configure FP8 quantization recipe
    print(f"Configuring FP8 quantization with format: {fp8_format}")
    
    # Define which modules to quantize
    targets = [
        "model.layers.*.self_attn.q_proj",
        "model.layers.*.self_attn.k_proj", 
        "model.layers.*.self_attn.v_proj",
        "model.layers.*.self_attn.o_proj",
        "model.layers.*.mlp.gate_proj",
        "model.layers.*.mlp.up_proj",
        "model.layers.*.mlp.down_proj",
    ]
    
    # Define modules to keep in higher precision
    ignore_patterns = []
    if preserve_critical_layers:
        ignore_patterns = [
            "model.embed_tokens",
            "lm_head",
            "model.norm",
            "model.layers.*.self_attn.rotary_emb",
            "model.layers.*.input_layernorm",
            "model.layers.*.post_attention_layernorm",
        ]
    
    # Create FP8 modifier
    fp8_modifier = FP8Modifier(
        targets=targets,
        fp8_format=fp8_format,
        calibration_batch_size=8,
        calibration_samples=num_calibration_samples,
        ignore=ignore_patterns,
        symmetric=True,
        static=True,  # Use static quantization for inference
    )
    
    # Create the quantization recipe
    recipe = [fp8_modifier]
    
    # Perform quantization
    print("Starting FP8 quantization...")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calibration_data,
        recipe=recipe,
        output_dir=output_path,
        save_format="native",  # Save in native format for vLLM
        save_precision="fp8",
        num_calibration_samples=num_calibration_samples,
        max_seq_length=max_seq_length,
        device_map="auto",
    )
    
    # Save quantization config for vLLM
    quantization_config = {
        "quant_method": "fp8",
        "fp8_format": fp8_format,
        "activation_scheme": "static",
        "quantize_kv_cache": quantize_kv_cache,
        "ignored_layers": ignore_patterns,
    }
    
    config_path = Path(output_path) / "quantization_config.json"
    with open(config_path, "w") as f:
        json.dump(quantization_config, f, indent=2)
    
    print(f"FP8 quantization complete! Model saved to: {output_path}")
    print(f"Quantization config saved to: {config_path}")
    
    # Print memory savings estimate
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    quantized_size = original_size / 2  # FP8 is roughly half the size of BF16
    print(f"\nEstimated model size reduction:")
    print(f"  Original (BF16): {original_size:.2f} GB")
    print(f"  Quantized (FP8): {quantized_size:.2f} GB")
    print(f"  Savings: {(1 - quantized_size/original_size) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Quantize Orpheus-TTS model to FP8")
    parser.add_argument(
        "--model-path",
        type=str,
        default="canopylabs/orpheus-3b-0.1-ft",
        help="Path to the base model"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./orpheus-3b-fp8",
        help="Path to save the quantized model"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer (defaults to model path)"
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=512,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for calibration"
    )
    parser.add_argument(
        "--fp8-format",
        type=str,
        choices=["fp8_e4m3", "fp8_e5m2"],
        default="fp8_e4m3",
        help="FP8 format to use (E4M3 for higher precision, E5M2 for wider range)"
    )
    parser.add_argument(
        "--no-quantize-kv-cache",
        action="store_true",
        help="Don't quantize KV cache"
    )
    parser.add_argument(
        "--quantize-all-layers",
        action="store_true",
        help="Quantize all layers including embeddings and lm_head"
    )
    
    args = parser.parse_args()
    
    quantize_model_to_fp8(
        model_path=args.model_path,
        output_path=args.output_path,
        tokenizer_path=args.tokenizer_path,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        fp8_format=args.fp8_format,
        quantize_kv_cache=not args.no_quantize_kv_cache,
        preserve_critical_layers=not args.quantize_all_layers,
    )


if __name__ == "__main__":
    main()
