"""Generate the deterministic audio fixture without third-party packages."""

import math
import pathlib
import struct
import wave


def main() -> None:
    output = pathlib.Path(__file__).parents[1] / "testdata" / "fixtures" / "sine-440hz-16k-mono.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    frame_count = sample_rate * 2
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = (
            struct.pack("<h", int(0.25 * 32767 * math.sin(2 * math.pi * 440 * i / sample_rate)))
            for i in range(frame_count)
        )
        wav.writeframes(b"".join(frames))
    print(f"wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
