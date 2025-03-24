import os
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download

# Initialize environment
load_dotenv()

# Log into huggingface in case access limited models need to be fetched
if huggingface_token := os.environ.get("HF_TOKEN"):
    login(huggingface_token)

# Set model names
snac_model_name = "hubertsiuzdak/snac_24khz"
orpheus_model_names = [
    "canopylabs/orpheus-3b-0.1-pretrained",
    "canopylabs/orpheus-3b-0.1-ft",
]

# Downlaoad snac model
snac_model_path = snapshot_download(
    repo_id=snac_model_name,
    allow_patterns=[
        "config.json",
        "pytorch_model.bin",
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
print(f"downloaded '{snac_model_name}' to '{snac_model_path}'.")

# Download orpheus models
for name in orpheus_model_names:
    path = snapshot_download(
        repo_id=name,
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
    print(f"downloaded '{name}' to '{path}'.")
