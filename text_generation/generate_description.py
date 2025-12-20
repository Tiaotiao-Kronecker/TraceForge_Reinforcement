#!/usr/bin/env python3
"""
Generate task descriptions for episodes using VLM.

Usage:
    # Single episode
    python generate_description.py --episode_dir <episode_path>

    # Batch mode: process all subfolders containing "images/"
    python generate_description.py --episode_dir <dataset_path>

    # Batch mode with cap
    python generate_description.py --episode_dir <dataset_path> --max_video 2

    # Skip existing
    python generate_description.py --episode_dir <dataset_path> --skip_existing
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Iterable, Optional, Tuple
import tqdm

# Use your existing generator (kept unchanged)
from text_generator import generate_task_description

# ----------------------------
# Natural sort for stable frame ordering
# ----------------------------
_NUM_RE = re.compile(r'(\d+)')

def _natural_key(p: Path) -> Tuple[int, list]:
    """
    Natural sort key that splits digits for correct numeric ordering.
    Also push names ending with '_last' to the very end.
    """
    stem = p.stem
    if stem.endswith('_last'):
        return (1, [float('inf')])
    parts = _NUM_RE.split(stem)
    key = [int(x) if x.isdigit() else x.lower() for x in parts]
    return (0, key)

# ----------------------------
# Frame selection
# ----------------------------
def list_images(images_dir: Path) -> List[Path]:
    """List image files in images_dir with common extensions and natural sort."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    files.sort(key=_natural_key)
    return files

def pick_five_frames(files: List[Path]) -> List[Path]:
    """
    Policy: Always include first & last; pick 3 evenly spaced frames from the middle.
    Deduplicate, preserve order, return at most 5.
    """
    if len(files) < 5:
        return False
    if len(files) == 5:
        return files

    first, last = files[0], files[-1]
    mids = files[1:-1]

    chosen = [first]
    if len(mids) <= 3:
        chosen += mids
    else:
        L = len(mids)
        idxs = [int((i + 1) * L / 4) for i in range(3)]  # 1/4, 2/4, 3/4
        chosen += [mids[i] for i in idxs]
    chosen.append(last)

    out, seen = [], set()
    for p in chosen:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:5]

# ----------------------------
# Episode discovery
# ----------------------------
def find_episodes(root: Path, images_subdir: str) -> List[Path]:
    """
    Return episode directories.
    - Single episode: if root/<images_subdir> exists, return [root]
    - Batch: else return all direct child dirs where <images_subdir> exists
    """
    images_dir = root / images_subdir
    if images_dir.is_dir():
        return [root]

    eps = []
    for child in sorted([d for d in root.iterdir() if d.is_dir()]):
        if (child / images_subdir).is_dir():
            eps.append(child)
    return eps

# ----------------------------
# Core processing
# ----------------------------
def process_episode(
    episode_dir: Path,
    images_subdir: str,
    provider: str,
    model_name: str,
    out_name: str,
    overwrite: bool = False
) -> Optional[Path]:
    """
    Process a single episode: select 5 frames and write JSON with 3 instructions.
    Returns output path if success, else None.
    """
    images_dir = episode_dir / images_subdir
    if not images_dir.is_dir():
        print(f"[SKIP] images dir missing: {images_dir}")
        return None

    out_path = episode_dir / out_name
    if out_path.exists() and not overwrite:
        print(f"[SKIP] exists: {out_path}")
        return out_path

    try:
        files = list_images(images_dir)
        picked = pick_five_frames(files)
        if picked is False:
            print(f"[SKIP] not enough frames: {episode_dir}")
            return None
        picked_strs = [str(p) for p in picked]

        resp = generate_task_description(
            picked_strs,
            provider=provider,
            model_name=model_name
        )

        # Accept bare 3-key or legacy wrapper
        candidate = resp.get("task_instructions", resp)
        just_three = {
            k: candidate[k] for k in ("instruction_1", "instruction_2", "instruction_3")
            if k in candidate
        }
        if not just_three:
            raise ValueError("Model response missing instruction_1/2/3")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(just_three, f, indent=2, ensure_ascii=False)

        print(f"[OK] {episode_dir.name} → {out_path}")
        return out_path

    except Exception as e:
        print(f"[ERR] {episode_dir}: {e}")
        return None

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate 3 short instructions for one or many episodes."
    )
    parser.add_argument("--episode_dir", type=str, required=True,
                        help="Path to a single episode OR a directory containing many episode subfolders.")
    parser.add_argument("--images_subdir", type=str, default="images",
                        help="Subdirectory name holding frames (default: images).")
    parser.add_argument("--provider", type=str, default="openai", choices=["google", "openai"],
                        help="Model provider (default: google).")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model name for the provider.")
    parser.add_argument("--out_name", type=str, default="three_instructions.json",
                        help="Output JSON filename saved inside each episode dir.")
    # Accept both --max_video and --max_videos for compatibility
    parser.add_argument("--max_video", "--max_videos", dest="max_video", type=int, default=None,
                        help="Maximum number of episodes to process (batch mode).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip episodes that already have the output file.")
    args = parser.parse_args()

    root = Path(args.episode_dir)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    episodes = find_episodes(root, args.images_subdir)
    if not episodes:
        raise FileNotFoundError(
            f"No episode found under {root} (looked for '{args.images_subdir}' subdirs)."
        )

    # Apply batch cap if requested
    if args.max_video is not None:
        episodes = episodes[: args.max_video]

    print(f"[INFO] Episodes to process: {len(episodes)}")
    processed = 0
    for ep in tqdm.tqdm(episodes):
        res = process_episode(
            ep,
            images_subdir=args.images_subdir,
            provider=args.provider,
            model_name=args.model,
            out_name=args.out_name,
            overwrite=not args.skip_existing
        )
        if res is not None:
            processed += 1

    print(f"[DONE] processed={processed} / total={len(episodes)}")

if __name__ == "__main__":
    main()
