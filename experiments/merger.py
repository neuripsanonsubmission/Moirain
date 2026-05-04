import argparse
import sys
import torch
import peft

from data import tokenizers
from models import model_moirain_multi
from experiments.utils import write_pkl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA (PEFT) weights into a base model and save a merged checkpoint."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the input .pth checkpoint file containing PEFT/LoRA weights.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer directory or file.",
    )
    return parser.parse_args()


def layers_to_peft(i):
    """Return the LoRA target module names for encoder block i."""
    base = f"trunk.encoder_na_{i}"
    return (
        f"{base}.self_mha.linear_q",
        f"{base}.self_mha.linear_kv",
        f"{base}.self_mha.linear_out",
        f"{base}.feed_forward.0",
        f"{base}.feed_forward.2",
    )


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint_path
    tokenizer_path = args.tokenizer_path

    print(f"[1/5] Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    for key in ("optimizer", "scheduler", "epoch", "step"):
        ckpt.pop(key, None)

    conf = ckpt["conf"]

    print(f"[2/5] Building tokenizer from: {tokenizer_path}")
    tokenizer = tokenizers.TokenizerRLMFlamingo(tokenizer_path)

    print("[3/5] Instantiating model …")
    model = model_moirain_multi.MainModel(conf.model, tokenizer)

    target_modules = [
        s
        for i in range(conf.model.na.num_blocks_na)
        for s in layers_to_peft(i)
    ]

    lora_config = peft.LoraConfig(
        r=conf.model.peft_rank,
        lora_alpha=conf.model.peft_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    model = peft.get_peft_model(model, lora_config)

    print("[4/5] Loading state dict …")
    result = model.load_state_dict(ckpt["model"], strict=True)

    unexpected = set(result.unexpected_keys)
    if unexpected:
        print(f"Unexpected keys found:\n  {unexpected}", file=sys.stderr)
        raise ValueError("Unexpected keys in the warm-up checkpoint!")

    print("[5/5] Merging LoRA weights and saving …")
    merged_model = model.merge_and_unload()

    checkpoint = {
        "model": merged_model.state_dict(),
        "conf": conf,
    }

    output_path = checkpoint_path.rsplit(".", 1)[0] + "_merged.pth"
    write_pkl(output_path, checkpoint, use_torch=True)
    print(f"Merged checkpoint saved to: {output_path}")


if __name__ == "__main__":
    main()