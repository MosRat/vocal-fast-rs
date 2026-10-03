"""Export the local RVC HuBERT checkpoint as a TensorRT-friendly ONNX graph.

The RVC frontend consumes only ``last_hidden_state``.  The final projection used
by v1 checkpoints stays outside this graph and is selected by the model
manifest.  The export deliberately keeps the waveform input dynamic and uses
opset 17, which is supported by TensorRT-RTX 1.6's ONNX parser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import HubertModel


class HubertFeatures(torch.nn.Module):
    def __init__(self, model: HubertModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model(input_values=waveform, return_dict=False)[0]


def main() -> None:
    # Windows hosts may keep the process stdout on cp936.  torch.onnx emits a
    # Unicode status marker while exporting; keep the exporter usable from a
    # normal PowerShell session.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/shared/hubert_base"))
    parser.add_argument(
        "--output", type=Path, default=Path("models/shared/hubert_base/hubert_features.onnx")
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--samples", type=int, default=32_000)
    args = parser.parse_args()

    model = HubertModel.from_pretrained(args.model, local_files_only=True)
    model.eval().float()
    wrapper = HubertFeatures(model).eval()
    waveform = torch.zeros(1, args.samples, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (waveform,),
            str(args.output),
            input_names=["waveform"],
            output_names=["features"],
            dynamic_axes={
                "waveform": {1: "samples"},
                "features": {1: "frames"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )
    print(args.output)


if __name__ == "__main__":
    main()
