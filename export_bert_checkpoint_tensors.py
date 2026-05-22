import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


STATE_TYPES = ["model_state", "master_weight", "exp_avg", "exp_avg_sq"]
CATEGORY_ORDER = ["key", "value", "query", "att_dense", "mlp_up", "mlp_down"]


def extract_all_layers_with_slices(checkpoint_dir, step_tag):
    """
    Dynamically scans and extracts weights and optimizer states for all BERT
    encoder layers while preserving the raw continuous FP32 streams.

    This function intentionally does not bin, clip, normalize, or otherwise
    alter the tensor values. The exported arrays remain a direct flattened view
    of the checkpoint payload so the downstream TensorAnalyzer binary can work
    with the original floating-point distributions.
    """
    weight_suffixes = {
        "key": "attention.self.key.weight",
        "value": "attention.self.value.weight",
        "query": "attention.self.query.weight",
        "att_dense": "attention.output.dense.weight",
        "mlp_up": "intermediate.dense.weight",
        "mlp_down": "output.dense.weight",
    }

    print(f"   Reconstructing model weights for {step_tag}...")
    model_state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag=step_tag)

    optim_file_path = Path(checkpoint_dir) / step_tag / "zero_pp_rank_0_mp_rank_00_optim_states.pt"
    print("   Extracting continuous optimizer streams from partition...")
    raw_optim_dict = torch.load(optim_file_path, map_location="cpu", weights_only=False)
    opt_dict = raw_optim_dict.get("optimizer_state_dict", {})

    base_state_groups = opt_dict.get("base_optimizer_state", {}).get("state", {})
    slice_mappings = opt_dict.get("param_slice_mappings", [])

    flat_exp_avg = base_state_groups[0].get("exp_avg") if 0 in base_state_groups else None
    flat_exp_avg_sq = base_state_groups[0].get("exp_avg_sq") if 0 in base_state_groups else None

    extracted_data = {}

    for layer_idx in range(12):
        for suffix_name, suffix_key in weight_suffixes.items():
            weight_key = f"bert.encoder.layer.{layer_idx}.{suffix_key}"

            if weight_key not in model_state_dict:
                continue

            model_tensor = model_state_dict[weight_key]
            master_key = f"master.{weight_key}"
            master_tensor = model_state_dict.get(master_key, model_tensor)

            extracted_data[weight_key] = {
                "layer": layer_idx,
                "category": suffix_name,
                "model_state": model_tensor.detach().cpu().float().numpy().flatten(),
                "master_weight": master_tensor.detach().cpu().float().numpy().flatten(),
                "exp_avg": np.array([], dtype=np.float32),
                "exp_avg_sq": np.array([], dtype=np.float32),
            }

            for group_maps in slice_mappings:
                if weight_key in group_maps:
                    fragment = group_maps[weight_key]
                    start = getattr(fragment, "start", fragment.get("start") if isinstance(fragment, dict) else None)
                    numel = getattr(fragment, "numel", fragment.get("numel") if isinstance(fragment, dict) else None)

                    if start is not None and numel is not None:
                        if flat_exp_avg is not None:
                            extracted_data[weight_key]["exp_avg"] = (
                                flat_exp_avg[start : start + numel].detach().cpu().float().numpy()
                            )
                        if flat_exp_avg_sq is not None:
                            extracted_data[weight_key]["exp_avg_sq"] = (
                                flat_exp_avg_sq[start : start + numel].detach().cpu().float().numpy()
                            )
                    break

    return extracted_data


def write_tensor_exports(extracted_data, output_dir):
    """
    Writes each tensor stream to a raw little-endian float32 binary file and
    records its location in a manifest CSV.

    The TensorAnalyzer binary deliberately consumes a very small interchange
    format: one CSV manifest plus raw `.bin` payloads. This keeps the C++ side
    independent from PyTorch and DeepSpeed while still preserving exact raw
    tensor values.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_root = output_dir / "tensors"
    tensor_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "tensor_key",
            "layer",
            "category",
            "state_type",
            "numel",
            "dtype",
            "relative_path",
        ])

        for tensor_key in sorted(extracted_data.keys()):
            entry = extracted_data[tensor_key]
            layer_idx = entry["layer"]
            category = entry["category"]

            for state_type in STATE_TYPES:
                array = entry.get(state_type)
                if array is None or array.size == 0:
                    continue

                state_dir = tensor_root / state_type / category
                state_dir.mkdir(parents=True, exist_ok=True)

                file_name = f"layer_{layer_idx:02d}.bin"
                relative_path = Path("tensors") / state_type / category / file_name
                file_path = output_dir / relative_path

                contiguous_array = np.ascontiguousarray(array.astype(np.float32, copy=False))
                contiguous_array.tofile(file_path)

                writer.writerow([
                    tensor_key,
                    layer_idx,
                    category,
                    state_type,
                    contiguous_array.size,
                    "float32",
                    relative_path.as_posix(),
                ])

    return manifest_path


def export_step(checkpoint_dir, step_tag, output_root):
    step_output_dir = output_root / step_tag
    print(f"Exporting raw tensors for {step_tag} into {step_output_dir}...")
    extracted_data = extract_all_layers_with_slices(checkpoint_dir, step_tag)
    manifest_path = write_tensor_exports(extracted_data, step_output_dir)
    print(f"   Wrote manifest: {manifest_path}")
    print(f"   Exported {len(extracted_data)} logical tensor groups")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export raw BERT checkpoint tensors into TensorAnalyzer's manifest-plus-binary format."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/home/olga/Documents/BERT/bert_sst2_checkpoints",
        help="Directory containing DeepSpeed checkpoint step folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/olga/Documents/TensorAnalyzer/exports",
        help="Directory where TensorAnalyzer-compatible exports will be written.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["step_500"],
        help="Checkpoint step tags to export.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_root = Path(args.output_dir)

    output_root.mkdir(parents=True, exist_ok=True)

    for step_tag in args.steps:
        step_path = checkpoint_dir / step_tag
        if not step_path.exists():
            print(f"Skipping missing step directory: {step_path}")
            continue

        try:
            export_step(checkpoint_dir, step_tag, output_root)
        except Exception as error:  # pragma: no cover - runtime data-path guard
            print(f"Failed to export {step_tag}: {error}")


if __name__ == "__main__":
    main()