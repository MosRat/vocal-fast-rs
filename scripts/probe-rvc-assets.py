"""Inspect real serialization, without executing pickle reducers."""
import collections
import json
import pickletools
import zipfile
from pathlib import Path

pth = Path('models/voices/default-v2/default.pth')
if zipfile.is_zipfile(pth):
    with zipfile.ZipFile(pth) as z:
        print('ZIP', z.namelist()[:10])
        b = z.read(next(n for n in z.namelist() if n.endswith('/data.pkl')))
        ops = list(pickletools.genops(b))
        print('opcodes', collections.Counter(o.name for o, _, _ in ops))
        print('globals', [a for o, a, _ in ops if o.name == 'GLOBAL'])
else:
    print('LEGACY')
    b = pth.read_bytes()
    for i in range(5):
        ops = list(pickletools.genops(b))
        print(i, 'opcodes', collections.Counter(o.name for o, _, _ in ops))
        print('globals', [a for o, a, _ in ops if o.name == 'GLOBAL'])
        print('prefix', [(o.name, str(a)[:120]) for o, a, _ in ops[:12]])
        b = b[ops[-1][2] + 1:]

import onnx
for path in Path('models/voices').rglob('*.onnx'):
    m = onnx.load(path, load_external_data=False)
    print(path, 'IR', m.ir_version, 'opsets', [(v.domain, v.version) for v in m.opset_import])
    print('meta', [(p.key, p.value) for p in m.metadata_props])
    print('IO', [(v.name, str(v.type).replace('\n', ' ')) for v in [*m.graph.input, *m.graph.output]])
    print('ops', sorted(set(n.op_type for n in m.graph.node)))
    print('weights', len(m.graph.initializer), [(t.name, list(t.dims), t.data_type) for t in m.graph.initializer[:8]])
    print('RVC anchors', [(t.name, list(t.dims)) for t in m.graph.initializer if any(k in t.name for k in ['emb_phone', 'emb_pitch', 'emb_g', 'conv_pre'])])
