from __future__ import annotations
import sqlite3, sys

db = sys.argv[1]
c = sqlite3.connect(db)
tables = [x[0] for x in c.execute("select name from sqlite_master where type='table' order by name")]
print("tables", tables)
for table in tables:
    cols = [x[1] for x in c.execute(f"pragma table_info({table})")]
    if "text" in cols and ("start" in cols or "startNs" in cols):
        print(table, cols)
for table in tables:
    cols = [x[1] for x in c.execute(f"pragma table_info({table})")]
    if "text" not in cols:
        continue
    start = "start" if "start" in cols else "startNs" if "startNs" in cols else None
    end = "end" if "end" in cols else "endNs" if "endNs" in cols else None
    dur = "duration" if "duration" in cols else "durationNs" if "durationNs" in cols else None
    if not start or not (end or dur):
        continue
    print("--", table)
    try:
        finish = end or f"({start}+{dur})"
        for row in c.execute(f"select text, {start}, ({finish}-{start}) from {table} where text like '%rvc.%' order by {start}"):
            print(row)
    except sqlite3.Error:
        pass

if "NVTX_EVENTS" in tables:
    rows = c.execute("select text, start, end from NVTX_EVENTS where text in ('rvc.hubert','rvc.rmvpe','rvc.frontend','rvc.synth') order by start").fetchall()
    print("-- nvtx summary (last three ranges per graph, ms)")
    for name in ("rvc.hubert", "rvc.rmvpe", "rvc.frontend", "rvc.synth"):
        vals = [(e-s)/1e6 for n,s,e in rows if n == name][-3:]
        print(name, vals, "mean", sum(vals)/len(vals) if vals else None)

if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
    print("-- top kernels by aggregate duration")
    query = """
    select s.value, sum(k.end-k.start)/1e6 as ms, count(*)
    from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
    group by k.demangledName order by ms desc limit 25
    """
    for row in c.execute(query):
        print(row)
