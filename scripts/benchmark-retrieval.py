"""Prepare/query real FAISS fixtures and benchmark the Rust CUDA retrieval test.

Build with `cargo test -p vocal-tensorrt --lib --no-run`, then pass its test
executable as --binary. Timings use CUDA events, with D2D input reset outside
the events and a single accuracy download after timing. No Python GPU code.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import faiss
import numpy as np


def reference(index, query):
    scores, ids = index.search(query, 8)
    vectors = index.reconstruct_n(0, index.ntotal)
    weights = np.zeros_like(scores)
    exact = (ids[:, 0] >= 0) & (scores[:, 0] <= 1e-12)
    weights[exact, 0] = 1
    normal = ~exact & (ids[:, 0] >= 0)
    weights[normal] = np.square(1 / scores[normal])
    weights[ids < 0] = 0
    sums = weights.sum(axis=1, keepdims=True)
    np.divide(weights, sums, out=weights, where=sums > 0)
    retrieved = (vectors[ids.clip(min=0)] * weights[..., None]).sum(axis=1)
    return np.where(sums > 0, 0.5 * retrieved + 0.5 * query, query)


def metrics(reference, actual):
    a, b = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    assert np.isfinite(b).all()
    delta = a - b
    return {"max_abs": float(np.abs(delta).max()),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "cosine": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))}


def fixtures(index, query):
    yield "real-34", index, query[:34].copy()
    yield "real-140", index, query[:140].copy()
    rng = np.random.default_rng(812)
    # Odd dimensions/partial blocks, short lists, exact matches and ties.
    for width, count in [(33, 3), (256, 65), (768, 257), (4096, 17)]:
        flat = faiss.IndexFlatL2(width)
        vectors = rng.standard_normal((count, width), dtype=np.float32)
        vectors[-1] = vectors[0]
        flat.add(vectors)
        q = rng.standard_normal((7, width), dtype=np.float32)
        q[0] = vectors[0]
        yield f"flat-{width}", flat, q
    width = 33
    vectors = rng.standard_normal((1024, width), dtype=np.float32)
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(width), width, 8)
    ivf.train(vectors)
    ivf.add(vectors[:19])
    q = rng.standard_normal((9, width), dtype=np.float32)
    q[0] = vectors[0]
    for probes in [1, 4, 8]:
        ivf.nprobe = probes
        yield f"ivf-{probes}", ivf, q
    # The nearest list is deliberately empty; blend must keep the query.
    coarse = faiss.IndexFlatL2(width)
    coarse.add(np.stack([np.zeros(width), np.ones(width) * 10]).astype(np.float32))
    empty = faiss.IndexIVFFlat(coarse, width, 2)
    empty.is_trained = True
    empty.add(np.ones((1, width), dtype=np.float32) * 10)
    yield "empty-list", empty, np.zeros((3, width), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="ratio=0 native debug JSON")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--real-only", action="store_true")
    args = parser.parse_args()
    faiss.omp_set_num_threads(1)
    index = faiss.read_index(str(args.index))
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8-sig"))
    query = np.asarray(snapshot["hubert_features"], np.float32).reshape(-1, index.d)
    if len(query) < 140:
        raise ValueError("snapshot must contain at least 140 raw HuBERT frames")
    reports = {}
    for name, idx, q in fixtures(index, query):
        if args.real_only and not name.startswith("real-"):
            continue
        root = args.output / name
        root.mkdir(parents=True, exist_ok=True)
        faiss.write_index(idx, str(root / "index.faiss"))
        q.tofile(root / "query.f32")
        expected = reference(idx, q)
        expected.tofile(root / "expected.f32")
        if not args.binary:
            continue
        env = os.environ | {"VOCAL_RETRIEVAL_BENCH": str(root.resolve()),
                            "VOCAL_RETRIEVAL_ITERATIONS": str(args.iterations)}
        result = subprocess.run([str(args.binary.resolve()),
                                 "retrieval::tests::benchmark_fixture", "--exact", "--ignored", "--nocapture"],
                                env=env, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        actual = np.fromfile(root / "actual.f32", np.float32).reshape(q.shape)
        parity = metrics(expected, actual)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        report = json.loads((root / "timings.json").read_text())
        report["faiss_parity"] = parity
        reports[name] = report
        print(name, json.dumps({"parity": parity, "timings": [
            {k: v for k, v in entry.items() if k != "samples_ms"} for entry in report["results"]]}), flush=True)
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
