from __future__ import annotations

import argparse
import sqlite3
import statistics


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in connection.execute(f"pragma table_info({table})")]


def ns(tx: int | float) -> float:
    return float(tx) / 1.0e6


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


parser = argparse.ArgumentParser(description="Summarize an Nsight Systems SQLite export")
parser.add_argument("database")
parser.add_argument(
    "--steady",
    type=int,
    default=10,
    help="number of final NVTX iterations used for steady summaries (default: 10)",
)
parser.add_argument(
    "--skip-runs",
    type=int,
    default=2,
    help="skip this many complete pipeline cycles before kernel/copy summaries (default: 2)",
)
args = parser.parse_args()

db = args.database
c = sqlite3.connect(db)
tables = [x[0] for x in c.execute("select name from sqlite_master where type='table' order by name")]
print("tables", tables)

steady_start = None
nvtx = []
if "NVTX_EVENTS" in tables:
    nvtx = c.execute(
        "select text, start, end from NVTX_EVENTS "
        "where text like 'rvc.%' and end is not null order by start"
    ).fetchall()
    graph_names = ("rvc.hubert", "rvc.rmvpe", "rvc.rmvpe.full", "rvc.frontend", "rvc.synth")
    def steady_values(values: list[float]) -> tuple[list[float], float]:
        """Reject capture/JIT outliers without hiding the raw range count."""
        if len(values) < 4:
            return values, 0.0
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        mad = statistics.median(deviations)
        # The first replay after TensorRT-RTX specialization can still be an
        # order of magnitude slower. Keep a generous 3x median floor while
        # rejecting that one-time work from the steady distribution.
        limit = max(median * 3.0, median + 6.0 * mad, 1.0e-6)
        filtered = [value for value in values if value <= limit]
        return (filtered or values), limit

    print("-- nvtx summary (last steady ranges, ms)")
    for name in graph_names:
        raw = [ns(e - s) for n, s, e in nvtx if n == name][-args.steady :]
        vals, limit = steady_values(raw)
        if vals:
            print(name, {
                "raw_count": len(raw),
                "steady_count": len(vals),
                "mean_ms": sum(vals) / len(vals),
                "median_ms": statistics.median(vals),
                "p95_ms": percentile(vals, 0.95),
                "max_ms": max(vals),
                "outlier_limit_ms": limit,
            })
    # The first graph range is normally the TensorRT capture/deserialize path.
    # Exclude it when aggregating kernels and copies so the report describes
    # replay, not plan loading.
    hubert_ranges = [(s, e) for n, s, e in nvtx if n == "rvc.hubert"]
    if len(hubert_ranges) > args.skip_runs:
        steady_start = hubert_ranges[args.skip_runs][0]


def show_activity(table: str, title: str, limit: int = 12) -> None:
    if table not in tables:
        return
    cols = table_columns(c, table)
    if "start" not in cols or "end" not in cols:
        return
    if table == "CUPTI_ACTIVITY_KIND_KERNEL":
        name_column = "demangledName"
        if name_column not in cols or "StringIds" not in tables:
            return
        cutoff = " and k.start >= ?" if steady_start is not None else ""
        query = (
            "select s.value, sum(k.end-k.start)/1e6 as ms, count(*) "
            "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName "
            f"where 1=1 {cutoff} group by k.demangledName order by ms desc limit ?"
        )
        values = ([steady_start] if steady_start is not None else []) + [limit]
        rows = c.execute(query, values).fetchall()
        if not rows:
            # Nsight exports can put NVTX and GPU activity on slightly
            # different timestamp domains.  Keep the kernel summary useful
            # instead of silently reporting an empty steady set.
            rows = c.execute(
                "select s.value, sum(k.end-k.start)/1e6 as ms, count(*) "
                "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName "
                "group by k.demangledName order by ms desc limit ?",
                [limit],
            ).fetchall()
            title += " (capture + replay; clock domains not alignable)"
    elif table == "CUPTI_ACTIVITY_KIND_MEMCPY":
        cutoff = " and start >= ?" if steady_start is not None else ""
        query = f"select sum(end-start)/1e6, count(*), sum(bytes) from {table} where 1=1 {cutoff}"
        rows = [c.execute(query, [steady_start] if steady_start is not None else []).fetchone()]
    elif table == "CUPTI_ACTIVITY_KIND_MEMSET":
        cutoff = " and start >= ?" if steady_start is not None else ""
        query = f"select sum(end-start)/1e6, count(*), sum(bytes) from {table} where 1=1 {cutoff}"
        rows = [c.execute(query, [steady_start] if steady_start is not None else []).fetchone()]
    else:
        return
    print(f"-- {title} (steady replay)")
    for row in rows:
        print(row)


def show_cuda_event_stage_pairs() -> None:
    """Report GPU stage durations from the pipeline's CUDA event pairs.

    NVTX ranges are host ranges around enqueueV3 and therefore do not measure
    the work executed by TensorRT's native CUDA graph.  RvcTrtPipeline records
    eight reusable CUDA events in creation order (start/end for HuBERT, RMVPE,
    frontend, and Synth).  Nsight exports their timestamps in
    CUPTI_ACTIVITY_KIND_CUDA_EVENT; pairing the recurring event streams gives
    a device-side stage view without downloading intermediate tensors.
    """
    if "CUPTI_ACTIVITY_KIND_CUDA_EVENT" not in tables:
        return
    groups = []
    for event_id, count, first in c.execute(
        "select eventId, count(*), min(timestamp) "
        "from CUPTI_ACTIVITY_KIND_CUDA_EVENT group by eventId order by eventId"
    ):
        if count < 4:
            continue
        timestamps = [
            row[0]
            for row in c.execute(
                "select timestamp from CUPTI_ACTIVITY_KIND_CUDA_EVENT "
                "where eventId = ? order by timestamp",
                [event_id],
            )
        ]
        groups.append((event_id, timestamps))
    if len(groups) < 8:
        return
    labels = ("hubert", "rmvpe", "frontend", "synth")
    print("-- CUDA event stage summary (device time, last steady ranges, ms)")
    for index, label in enumerate(labels):
        # CUDA streams can overlap, so timestamp order is not event creation
        # order. The pipeline creates its reusable start/end events in the
        # HuBERT/RMVPE/frontend/Synth order; preserve eventId order here.
        starts = groups[index * 2][1]
        ends = groups[index * 2 + 1][1]
        values = [ns(end - start) for start, end in zip(starts, ends)]
        values = values[-args.steady :]
        if values:
            print(label, {
                "count": len(values),
                "mean_ms": sum(values) / len(values),
                "median_ms": statistics.median(values),
                "p95_ms": percentile(values, 0.95),
                "min_ms": min(values),
                "max_ms": max(values),
            })


show_cuda_event_stage_pairs()


show_activity("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy")
show_activity("CUPTI_ACTIVITY_KIND_MEMSET", "memset")
show_activity("CUPTI_ACTIVITY_KIND_KERNEL", "top kernels")
