#!/usr/bin/env python
"""Run the Phase 2 retrieval MVP on an exported dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_retrieval.retrieval.pipeline import run_leave_one_out_retrieval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v1_reach_target",
        help="Exported dataset root.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="pose_descriptor",
        choices=[
            "pose_descriptor",
            "rgb_histogram",
            "global_color",
            "geometry_only",
            "random",
            "uni3d",
            "ptv3",
            "point_transformer_v3",
        ],
        help="Episode encoder to use.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of neighbors to return.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "retrieval",
        help="Directory where retrieval outputs are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_leave_one_out_retrieval(
        args.dataset_root,
        encoder_name=args.encoder,
        k=args.k,
    )

    output_dir = args.output_dir / args.encoder
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for query_id, matches in result.matches.items():
        for match in matches:
            manifest_rows.append(
                {
                    "query_episode_id": query_id,
                    "candidate_episode_id": match.candidate_episode_id,
                    "rank": match.rank,
                    "score": match.score,
                }
            )

    pd.DataFrame(manifest_rows).to_csv(output_dir / "retrieval_results.csv", index=False)
    (output_dir / "retrieval_results.json").write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "encoder": args.encoder,
                "k": args.k,
                "results": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("Phase 2: Retrieval MVP")
    print("=" * 80)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Encoder: {args.encoder}")
    print(f"Results written to: {output_dir}")
    print("\nTop-1 neighbors per query:")
    for query_id, matches in result.matches.items():
        if not matches:
            print(f"  {query_id}: no candidates")
            continue
        match = matches[0]
        print(f"  {query_id} -> {match.candidate_episode_id}  score={match.score:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
