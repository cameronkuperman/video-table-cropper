#!/usr/bin/env python3
"""
AutoLabeler - two modes:

  python main.py --process   download videos from Drive, extract frames,
                             draw overlays, crop tables → upload to unlabeled/

  python main.py --label     start Flask UI at localhost:8080 to label
                             images and move them to clean/dirty/occupied/
"""

from __future__ import annotations

import argparse
import os
import sys
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
    parser.add_argument("--port", type=int, default=8080, help="Port for --label UI (default: 8080)")
    args = parser.parse_args()

    if args.process:
        cmd_process()
    elif args.label:
        cmd_label(args.port)


if __name__ == "__main__":
    main()
