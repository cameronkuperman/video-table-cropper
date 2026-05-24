#!/usr/bin/env python3
"""
AutoLabeler modes:

  python main.py --process   download videos from Drive, extract frames,
                             draw overlays, crop tables → upload to unlabeled/

  python main.py --label     start Flask UI at localhost:8080 to label
                             images and move them to clean/dirty/occupied/

  python main.py --preprocess-until-empty --sources reolink
                             drain deploy worker preprocessing and exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from env_loader import load_local_env

load_local_env()


def cmd_process() -> None:
    from drive_client import DriveClient, DriveClientError
    from processor import run_processor

    root_id = os.environ.get("DRIVE_PROJECT_ROOT_FOLDER_ID", "").strip()
    if not root_id:
        print("Error: DRIVE_PROJECT_ROOT_FOLDER_ID is not set in .env")
        sys.exit(1)

    # Find the table JSON (prefer approved_table_rectangles.json)
    base = Path(__file__).parent
    tables_json = base / "approved_table_rectangles.json"
    if not tables_json.exists():
        tables_json = base / "approved_tables.json"
    if not tables_json.exists():
        print("Error: approved_table_rectangles.json (or approved_tables.json) not found")
        sys.exit(1)

    print(f"Using table config: {tables_json.name}")

    try:
        client = DriveClient()
        run_processor(root_id, tables_json, client)
    except DriveClientError as e:
        print(f"Drive error: {e}")
        sys.exit(1)


def _root_id_or_exit() -> str:
    root_id = os.environ.get("DRIVE_PROJECT_ROOT_FOLDER_ID", "").strip()
    if not root_id:
        print("Error: DRIVE_PROJECT_ROOT_FOLDER_ID is not set in .env")
        sys.exit(1)
    return root_id


def _tables_json_or_exit() -> Path:
    base = Path(__file__).parent
    tables_json = base / "approved_table_rectangles.json"
    if not tables_json.exists():
        tables_json = base / "approved_tables.json"
    if not tables_json.exists():
        print("Error: approved_table_rectangles.json (or approved_tables.json) not found")
        sys.exit(1)
    return tables_json


def _effective_preprocess_sources(cli_sources: str) -> str:
    override = os.environ.get("PREPROCESS_SOURCES", "").strip().lower()
    if not override:
        return cli_sources
    if override not in {"all", "video", "reolink"}:
        print("Error: PREPROCESS_SOURCES must be one of: all, video, reolink")
        sys.exit(1)
    return override


def cmd_preprocess_until_empty(sources: str) -> None:
    from drive_client import DriveClient, DriveClientError

    sources = _effective_preprocess_sources(sources)
    root_id = _root_id_or_exit()
    client = DriveClient()
    summary: dict[str, object] = {"sources": sources}

    try:
        if sources in {"all", "video"}:
            from processor import run_processor

            video_summary = run_processor(root_id, _tables_json_or_exit(), client)
            summary["video"] = asdict(video_summary)

        if sources in {"all", "reolink"}:
            from app import drain_reolink_preprocessing

            summary["reolink"] = drain_reolink_preprocessing(client)
    except DriveClientError as e:
        print(f"Drive error: {e}")
        sys.exit(1)

    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_label(port: int) -> None:
    from app import run_label_ui
    run_label_ui(port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoLabeler: process videos or label cropped table images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--process", action="store_true", help="Process videos from Drive")
    group.add_argument("--label", action="store_true", help="Start labeling UI")
    group.add_argument(
        "--preprocess-until-empty",
        action="store_true",
        help="Drain video and/or Reolink preprocessing queues, then exit",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port for --label UI (default: 8080)")
    parser.add_argument(
        "--sources",
        choices=("all", "video", "reolink"),
        default="all",
        help="Sources for --preprocess-until-empty (default: all)",
    )
    args = parser.parse_args()

    if args.process:
        cmd_process()
    elif args.label:
        cmd_label(args.port)
    elif args.preprocess_until_empty:
        cmd_preprocess_until_empty(args.sources)


if __name__ == "__main__":
    main()
