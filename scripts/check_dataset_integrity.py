#!/usr/bin/env python3
"""Check for missing MIDI labels in PianoVAM dataset."""

import os
from pathlib import Path

# Get dataset paths from environment
DATASET_ROOT = Path(os.environ.get('PIANOVAM_ROOT', '/raid_storage/data_achatzigiannis/PianoVAM_v1.0'))
VIDEO_DIR = DATASET_ROOT / 'Video'
MIDI_DIR = DATASET_ROOT / 'MIDI'

print(f"\n{'='*80}")
print("PianoVAM Dataset Integrity Check")
print(f"{'='*80}")
print(f"Dataset root: {DATASET_ROOT}")
print(f"Video dir: {VIDEO_DIR}")
print(f"MIDI dir: {MIDI_DIR}\n")

# Get video and MIDI files
video_files = set(f.stem for f in sorted(VIDEO_DIR.glob('*.mp4')))
midi_files = set(f.stem for f in sorted(MIDI_DIR.glob('*.mid')))

print(f"Total videos: {len(video_files)}")
print(f"Total MIDI files: {len(midi_files)}")

# Find missing MIDI labels
missing_midi = video_files - midi_files
print(f"\n{'='*80}")
print(f"Videos WITHOUT MIDI labels: {len(missing_midi)}")
print(f"{'='*80}")

if missing_midi:
    print("\nMissing MIDI files:")
    for vid_id in sorted(missing_midi):
        print(f"  ✗ {vid_id}")
        video_path = VIDEO_DIR / f'{vid_id}.mp4'
        print(f"    Video: {video_path}")
        print(f"    MIDI: {MIDI_DIR / f'{vid_id}.mid'} (MISSING)")
    
    print(f"\n{'='*80}")
    print("ACTION REQUIRED:")
    print(f"{'='*80}")
    print(f"\nOption 1: Remove videos without MIDI labels")
    print("Run this command to remove them:")
    print()
    for vid_id in sorted(missing_midi):
        print(f'  rm "{VIDEO_DIR / f"{vid_id}.mp4"}"')
    
    print(f"\n\nOption 2: Generate a simple cleanup script")
    cleanup_script = Path("cleanup_videos.sh")
    with open(cleanup_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Remove videos without MIDI labels\n\n")
        for vid_id in sorted(missing_midi):
            f.write(f'rm "{VIDEO_DIR / f"{vid_id}.mp4"}"\n')
    
    print(f"  Created {cleanup_script}")
    print(f"  Run: chmod +x {cleanup_script} && ./{cleanup_script}")
    
else:
    print("\n✓ All videos have corresponding MIDI labels!")
    print("Dataset is complete and consistent.")

# Find orphaned MIDI files (MIDI without video)
orphaned_midi = midi_files - video_files
if orphaned_midi:
    print(f"\n{'='*80}")
    print(f"Orphaned MIDI files (no video): {len(orphaned_midi)}")
    print(f"{'='*80}")
    for mid_id in sorted(orphaned_midi)[:10]:  # Show first 10
        print(f"  ✗ {mid_id}")
    if len(orphaned_midi) > 10:
        print(f"  ... and {len(orphaned_midi) - 10} more")

print(f"\n{'='*80}\n")
