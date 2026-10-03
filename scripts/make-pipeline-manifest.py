import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--synth', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()
d = json.loads(a.source.read_text())
d['synth_plan'] = str(a.synth)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(d, indent=2) + '\n')
