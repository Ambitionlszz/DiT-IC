"""Small CUDA kernel/precision check; does not load weights or start training."""

import argparse
import json
from pathlib import Path


def check_gpu(torch, index, expected):
    device = torch.device(f"cuda:{index}")
    capability = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    if expected == "5090" and ("5090" not in name or capability != (12, 0)):
        raise RuntimeError(f"expected RTX 5090 sm_120, got {name} {capability}")
    if expected == "v100" and ("V100" not in name or capability != (7, 0)):
        raise RuntimeError(f"expected V100 sm_70, got {name} {capability}")
    arch = f"sm_{capability[0]}{capability[1]}"
    if arch not in torch.cuda.get_arch_list():
        raise RuntimeError(f"{name} needs {arch}; installed wheel contains {torch.cuda.get_arch_list()}")
    with torch.cuda.device(device), torch.enable_grad():
        native_bf16 = capability[0] >= 8 and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if native_bf16 else torch.float16
        image = torch.randn(1, 8, 32, 32, device=device, requires_grad=True)
        conv = torch.nn.Conv2d(8, 8, 3, padding=1).to(device)
        query = torch.randn(1, 2, 32, 16, device=device, requires_grad=True)
        with torch.autocast("cuda", dtype=dtype):
            output = conv(image)
            attention = torch.nn.functional.scaled_dot_product_attention(query, query, query)
            loss = output.float().square().mean() + attention.float().square().mean()
        loss.backward()
        values = (loss, image.grad, query.grad, conv.weight.grad)
        if not all(value is not None and bool(torch.isfinite(value).all()) for value in values):
            raise RuntimeError(f"nonfinite forward/backward on {name}")
        torch.cuda.synchronize(device)
    return {"index": index, "name": name, "capability": list(capability),
            "precision": "bf16" if native_bf16 else "fp16+GradScaler in training",
            "convolution_attention_backward": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", choices=("auto", "v100", "5090"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"status": "failed", "expected_gpu": args.expect, "gpus": []}
    try:
        import torch
        report.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                      compiled_architectures=torch.cuda.get_arch_list())
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check the driver and the selected Python environment")
        for index in range(torch.cuda.device_count()):
            report["gpus"].append(check_gpu(torch, index, args.expect))
        report["status"] = "passed"
    except Exception as error:
        report["reason"] = f"{type(error).__name__}: {error}"
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
