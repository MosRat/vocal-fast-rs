"""Create a deterministic Synth-only VIN fixture from frontend debug output."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b"VFRTEST1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", type=Path, required=True)
    ap.add_argument("--vfrm", type=Path, default=ROOT / "target/tsukuyomi-canonical.vfr")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--native-deterministic",
        action="store_true",
        help="use the exact latent/noise sequence seeded by RvcTrtPipeline",
    )
    args = ap.parse_args()
    debug = json.loads(args.debug.read_text(encoding="utf-8"))
    feats = np.asarray(debug["rvc_features"], dtype="<f4")
    rvc_frames = int(debug.get("rvc_frames", len(debug["pitch"])))
    feats = feats.reshape(1, rvc_frames, -1)
    pitch = np.asarray(debug["pitch"], dtype="<i4").reshape(1, rvc_frames)
    pitchf = np.asarray(debug.get("aligned_pitchf", debug["pitchf"]), dtype="<f4").reshape(1, rvc_frames)
    raw = args.vfrm.read_bytes()
    metadata_len = struct.unpack_from("<Q", raw, 16)[0]
    config = json.loads(raw[64 : 64 + metadata_len])
    c = config["metadata"]["rvc_config"]
    latent = int(c["latent_channels"])
    upsample = int(np.prod(c["upsample_rates"], dtype=np.int64))
    if args.native_deterministic:
        rnd = (0.01 * np.sin(np.arange(latent * rvc_frames, dtype=np.float32) * 0.017)) \
            .reshape(1, latent, rvc_frames).astype(np.float32)
        noise = (0.001 * np.sin(np.arange(rvc_frames * upsample, dtype=np.float32) * 0.013)) \
            .reshape(1, 1, rvc_frames * upsample).astype(np.float32)
    else:
        rng = np.random.default_rng(args.seed)
        rnd = rng.standard_normal((1, latent, rvc_frames), dtype=np.float32)
        noise = rng.standard_normal((1, 1, rvc_frames * upsample), dtype=np.float32)
    p_len = min(int(debug.get("pitch_frames", rvc_frames)), rvc_frames)
    sid = 0
    with args.output.open("wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<6I", rvc_frames, feats.shape[2], latent, upsample, p_len, sid))
        out.write(np.ascontiguousarray(feats).tobytes())
        out.write(np.asarray([p_len, sid], dtype="<i4").tobytes())
        out.write(np.ascontiguousarray(rnd).tobytes())
        out.write(np.ascontiguousarray(pitch).tobytes())
        out.write(np.ascontiguousarray(pitchf).tobytes())
        out.write(np.ascontiguousarray(noise).tobytes())
    print(json.dumps({"output": str(args.output), "frames": rvc_frames, "p_len": p_len,
                      "feature_width": feats.shape[2], "latent": latent, "upsample": upsample,
                      "native_deterministic": args.native_deterministic}, indent=2))


if __name__ == "__main__":
    main()
