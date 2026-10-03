import wave, numpy as np
from pathlib import Path
src=Path('testdata/fixtures/sine-440hz-16k-mono.wav'); out=Path('target/compare/sine-11200-input.f32')
with wave.open(str(src),'rb') as w:
    x=np.frombuffer(w.readframes(11200),dtype='<i2').astype(np.float32)/32768.0
out.write_bytes(x.tobytes()); print(out, x.size)
