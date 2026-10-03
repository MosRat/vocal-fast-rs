import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils.parametrize as parametrize
import sys
sys.path.insert(0, "ref_repos/Applio")
from rvc.lib.algorithm.synthesizers import Synthesizer


def read(path):
    b = Path(path).read_bytes()
    assert b[:8] == b"VFRM\0\0\0\1"
    metadata_len, count, directory_len, data_offset = struct.unpack_from("<QQQQ", b, 16)
    metadata = b[64:64 + metadata_len]
    at = 64 + metadata_len
    names = []
    for _ in range(count):
        name_len, dtype, rank, _ = struct.unpack_from("<IBBH", b, at)
        at += 8
        name = b[at:at + name_len].decode(); at += name_len
        shape = struct.unpack_from("<" + "Q" * rank, b, at); at += rank * 8
        off, size = struct.unpack_from("<QQ", b, at); at += 16
        names.append((name, shape, dtype, off, size))
    return names


cfg = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3, [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000]
net = Synthesizer(*cfg, use_f0=True, text_enc_hidden_dim=768).float().eval()
for module in net.modules():
    if parametrize.is_parametrized(module, "weight"):
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
state = net.state_dict()
names = read("target/tsukuyomi-canonical.vfr")
canonical = {x[0] for x in names}
state_names = set(state)
print("canonical", len(canonical), "state", len(state_names))
print("missing in state", sorted(canonical - state_names))
print("not in canonical", len(state_names - canonical), sorted(state_names - canonical)[:30])
for name, shape, dtype, off, size in names:
    if name in state:
        if tuple(state[name].shape) != shape and name == "enc_p.emb_phone.weight":
            print(name, "canonical", shape, "state", tuple(state[name].shape), "transpose expected")
        elif tuple(state[name].shape) != shape:
            print(name, "canonical", shape, "state", tuple(state[name].shape))
