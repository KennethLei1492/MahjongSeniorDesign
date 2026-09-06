"""Export a trained checkpoint for deployment on the NexArm host computer.

TorchScript is the primary format (runs on the Raspberry-Pi-class host that
drives the arm); ONNX is also emitted for alternative runtimes (e.g.
onnxruntime, or conversion toward the K230 vision module's nncase toolchain).

Usage:
    python -m rl.export --checkpoint checkpoints/latest.pt --out deploy/
"""
import argparse
import os

import torch
from .model import build_model
from .encoder import NUM_PLANES, HISTORY_LEN, EVENT_DIM
from hk_mahjong.actions import NUM_ACTIONS


def export(checkpoint, out_dir, name="mahjong_policy"):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(checkpoint, map_location="cpu")
    # Architecture comes from the checkpoint itself (missing on legacy
    # small-model checkpoints, whose defaults build_model already carries),
    # so any variant - small, big, score-reward - exports correctly.
    cfg = ckpt.get("config") or {}
    model = build_model("cpu", **cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ex_planes = torch.zeros(1, NUM_PLANES, 4, 34)
    ex_hist = torch.zeros(1, HISTORY_LEN, EVENT_DIM)
    ex_mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)

    ts = torch.jit.trace(model, (ex_planes, ex_hist, ex_mask))
    ts_path = os.path.join(out_dir, name + ".ts.pt")
    ts.save(ts_path)
    print("TorchScript ->", ts_path)

    onnx_path = os.path.join(out_dir, name + ".onnx")
    torch.onnx.export(model, (ex_planes, ex_hist, ex_mask), onnx_path,
                      input_names=["planes", "history", "mask"],
                      output_names=["logits", "value"], opset_version=17)
    print("ONNX ->", onnx_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/latest.pt")
    ap.add_argument("--out", default="deploy")
    ap.add_argument("--name", default="mahjong_policy",
                    help="output basename, e.g. champion_win or "
                         "challenger_score, so multiple models can coexist "
                         "in deploy/")
    args = ap.parse_args()
    export(args.checkpoint, args.out, args.name)
