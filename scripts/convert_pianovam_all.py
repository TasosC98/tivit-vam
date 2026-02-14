from __future__ import annotations
from pathlib import Path
import argparse
import subprocess
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--frames", type=int, default=96)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--resize-h", type=int, default=180)
    p.add_argument("--resize-w", type=int, default=1536)
    p.add_argument("--max", type=int, default=0, help="convert only first N videos (0=all)")
    args = p.parse_args()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.MP4")))
    if args.max and args.max > 0:
        vids = vids[:args.max]

    if not vids:
        print(f"No mp4 files found in {video_dir}")
        return 2

    script = Path(__file__).resolve().parent / "convert_videos_to_hdf5.py"

    done = 0
    skipped = 0
    for v in vids:
        out_h5 = out_dir / (v.stem + ".h5")
        if out_h5.exists() and out_h5.stat().st_size > 0:
            skipped += 1
            continue

        cmd = [
            sys.executable, str(script),
            "--video", str(v),
            "--out-dir", str(out_dir),
            "--frames", str(args.frames),
            "--stride", str(args.stride),
            "--resize", str(args.resize_h), str(args.resize_w),
        ]
        print("RUN:", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"FAILED on {v} (return code {r.returncode}). Stopping.")
            return r.returncode
        done += 1

    print(f"Finished. converted={done}, skipped={skipped}, total={len(vids)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())