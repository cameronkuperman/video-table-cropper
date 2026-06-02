#!/usr/bin/env python3
"""Audit and optionally soft-trash ScreenRecord true-10 raw folders.

The script is intentionally dry-run first. It only soft-trashes raw
10frametrue folders, never generated/labeled crop folders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("LABEL_READY_MAINTAINER_ON_STARTUP", "0")
os.environ.setdefault("LABEL_DRAIN_ON_STARTUP", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import app as label_app  # noqa: E402
from drive_client import DriveClient  # noqa: E402

CONFIRM_TEXT = "TRASH_TRUE_TEN_RAW"
GENERATED_BUCKET = "generated"
LABELED_BUCKETS = {"clean", "dirty", "occupied", "label_later", "discarded"}


def _jsonl_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("/private/tmp") / f"screenrecord_true_ten_reconcile_{stamp}.jsonl"


def _channel_filter(value: str | None) -> str | None:
    if not value:
        return None
    return label_app._normalize_reolink_channel_code(str(value))  # noqa: SLF001


def _load_metadata(client: DriveClient, folder_id: str) -> dict[str, Any] | None:
    metadata_item = client.find_file_by_name(folder_id, "metadata.json")
    return label_app._load_json_file_from_drive(client, metadata_item)  # noqa: SLF001


def _artifact_table_keys(metadata: dict[str, Any]) -> set[str]:
    values = {
        metadata.get("table_camera_crops_id"),
        metadata.get("supabase_table_id"),
        metadata.get("table_id"),
        metadata.get("table_label"),
    }
    table = metadata.get("table")
    if isinstance(table, dict):
        values.update({table.get("id"), table.get("label")})
    return {str(value).strip() for value in values if str(value or "").strip()}


def _scan_generated_artifacts(client: DriveClient, context: label_app.QueueContext) -> dict[str, Any]:
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_raw_table: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    scanned = 0
    metadata_found = 0

    parents: list[tuple[str, str]] = []
    for folder_id in label_app._context_input_folder_ids(context):  # noqa: SLF001
        parents.append((GENERATED_BUCKET, folder_id))
    for bucket in label_app.LABEL_DESTINATIONS:
        parent_id = context.folder_ids.get(bucket)
        if parent_id:
            parents.append((bucket, parent_id))

    seen_parent_ids: set[str] = set()
    for bucket, parent_id in parents:
        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        for folder in client.list_folders(parent_id, fields="id,name,mimeType,parents,appProperties,modifiedTime"):
            folder_id = str(folder.get("id") or "")
            folder_name = str(folder.get("name") or "")
            if not folder_id or not folder_name:
                continue
            scanned += 1
            record = {
                "folder_id": folder_id,
                "folder_name": folder_name,
                "bucket": bucket,
                "parent_id": parent_id,
            }
            by_name[folder_name].append(record)
            metadata = _load_metadata(client, folder_id)
            if not metadata:
                continue
            metadata_found += 1
            identity = label_app._metadata_identity(metadata)  # noqa: SLF001
            if identity:
                by_identity[identity].append(record)
            raw_folder_id = str(metadata.get("raw_folder_id") or "").strip()
            raw_folder_name = str(metadata.get("raw_folder_name") or "").strip()
            for raw_key in {raw_folder_id, raw_folder_name} - {""}:
                for table_key in _artifact_table_keys(metadata):
                    by_raw_table[(raw_key, table_key)].append(record)

    return {
        "by_identity": by_identity,
        "by_name": by_name,
        "by_raw_table": by_raw_table,
        "scanned": scanned,
        "metadata_found": metadata_found,
    }


def _resolve_expected(
    client: DriveClient,
    context: label_app.QueueContext,
    raw_folder: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[str | None, int, dict[str, Any] | None, list[tuple[str, list, tuple[int, int, int, int], list]]]:
    raw_name = str(raw_folder.get("name") or "")
    supabase_client = label_app._get_supabase_crop_client()  # noqa: SLF001
    if supabase_client.enabled:
        mapped = label_app._resolve_supabase_crop_tables(raw_name, metadata, site_key=context.site_key)  # noqa: SLF001
        if mapped is None:
            return "unsafe_supabase_unresolved", 0, None, []
        channel_number, camera, table_polygons = mapped
        return None, channel_number, camera, table_polygons

    mapped = label_app._mapped_camera_tables_for_screenrecord_folder(  # noqa: SLF001
        raw_name,
        metadata,
        site_key=context.site_key,
        client=client,
    )
    if mapped is None:
        return "missing_mapping", 0, None, []
    channel_number, camera, table_polygons = mapped
    return None, channel_number, camera, table_polygons


def _matched_artifacts(
    *,
    raw_folder: dict[str, Any],
    metadata: dict[str, Any],
    table_id: str,
    camera: dict[str, Any],
    label_source: label_app.LabelSource,
    artifact_index: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_name = str(raw_folder.get("name") or "")
    raw_id = str(raw_folder.get("id") or "")
    table_metadata = label_app._camera_table_metadata(camera, table_id)  # noqa: SLF001
    identity = label_app._artifact_identity(raw_name, table_metadata, metadata)  # noqa: SLF001
    legacy_name = label_app._derived_reolink_folder_name(raw_name, table_id)  # noqa: SLF001
    prefixed_name = label_app._apply_source_prefix(legacy_name, label_source)  # noqa: SLF001

    matches: dict[str, dict[str, Any]] = {}
    for record in artifact_index["by_identity"].get(identity, []):
        matches[record["folder_id"]] = record
    for name in (legacy_name, prefixed_name):
        for record in artifact_index["by_name"].get(name, []):
            matches[record["folder_id"]] = record
    for table_key in {
        str(table_metadata.get("table_camera_crops_id") or "").strip(),
        str(table_metadata.get("table_id") or "").strip(),
        str(metadata.get("table_id") or "").strip(),
    } - {""}:
        for raw_key in (raw_id, raw_name):
            if not raw_key:
                continue
            for record in artifact_index["by_raw_table"].get((raw_key, table_key), []):
                matches[record["folder_id"]] = record
    return list(matches.values())


def _decision(
    *,
    mapping_error: str | None,
    expected_count: int,
    generated_ratio: float,
    labeled_ratio: float,
    min_generated_ratio: float,
    require_labeled: bool,
) -> str:
    if mapping_error:
        return mapping_error
    if expected_count <= 0:
        return "missing_expected_crops"
    if require_labeled:
        if labeled_ratio >= min_generated_ratio:
            return "eligible_labeled_threshold"
        if generated_ratio >= min_generated_ratio:
            return "generated_but_not_labeled_threshold"
        return "partial"
    if generated_ratio >= min_generated_ratio:
        return "eligible_generated_threshold"
    if generated_ratio > 0:
        return "partial"
    return "missing"


def reconcile(args: argparse.Namespace) -> int:
    if args.mode == "trash" and args.confirm != CONFIRM_TEXT:
        raise SystemExit(f"--mode trash requires --confirm {CONFIRM_TEXT}")

    client = DriveClient()
    context = label_app._resolve_queue_context(client, label_app.REOLINK_SOURCE, args.site)  # noqa: SLF001
    channel_filter = _channel_filter(args.channel)
    output_path = Path(args.output) if args.output else _jsonl_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_index = _scan_generated_artifacts(client, context)
    raw_folders = label_app._list_screenrecord_true_ten_folders(client, context)  # noqa: SLF001
    label_source = label_app._resolve_label_source(context.source, context.site_key)  # noqa: SLF001

    counts: dict[str, int] = defaultdict(int)
    processed = 0
    with output_path.open("w", encoding="utf-8") as out:
        header = {
            "type": "summary_header",
            "site": args.site,
            "mode": args.mode,
            "min_generated_ratio": args.min_generated_ratio,
            "require_labeled": args.require_labeled,
            "generated_artifacts_scanned": artifact_index["scanned"],
            "generated_artifacts_with_metadata": artifact_index["metadata_found"],
        }
        out.write(json.dumps(header, sort_keys=True) + "\n")

        for raw_folder in raw_folders:
            if args.limit is not None and processed >= args.limit:
                break
            raw_name = str(raw_folder.get("name") or "")
            metadata = _load_metadata(client, str(raw_folder["id"]))
            if not metadata:
                raw_channel = label_app._screenrecord_channel_code_from_metadata(raw_name, {})  # noqa: SLF001
                if channel_filter and raw_channel != channel_filter:
                    continue
                decision = "missing_metadata"
                record = {
                    "type": "raw_folder",
                    "raw_folder_id": raw_folder.get("id"),
                    "raw_folder_name": raw_name,
                    "channel_code": raw_channel,
                    "expected_count": 0,
                    "generated_found_count": 0,
                    "labeled_found_count": 0,
                    "generated_ratio": 0,
                    "labeled_ratio": 0,
                    "decision": decision,
                    "trashed": False,
                }
            else:
                raw_channel = label_app._screenrecord_channel_code_from_metadata(raw_name, metadata)  # noqa: SLF001
                if channel_filter and raw_channel != channel_filter:
                    continue
                mapping_error, _channel_number, camera, table_polygons = _resolve_expected(client, context, raw_folder, metadata)
                expected_count = len(table_polygons)
                generated_found = 0
                labeled_found = 0
                matched: list[dict[str, Any]] = []
                for table_id, *_rest in table_polygons:
                    table_matches = _matched_artifacts(
                        raw_folder=raw_folder,
                        metadata=metadata,
                        table_id=table_id,
                        camera=camera or {},
                        label_source=label_source,
                        artifact_index=artifact_index,
                    )
                    if table_matches:
                        generated_found += 1
                        matched.extend(table_matches)
                    if any(match.get("bucket") in LABELED_BUCKETS for match in table_matches):
                        labeled_found += 1

                generated_ratio = generated_found / expected_count if expected_count else 0.0
                labeled_ratio = labeled_found / expected_count if expected_count else 0.0
                decision = _decision(
                    mapping_error=mapping_error,
                    expected_count=expected_count,
                    generated_ratio=generated_ratio,
                    labeled_ratio=labeled_ratio,
                    min_generated_ratio=args.min_generated_ratio,
                    require_labeled=args.require_labeled,
                )
                trashed = False
                if args.mode == "trash" and decision in {"eligible_generated_threshold", "eligible_labeled_threshold"}:
                    client.trash_file(str(raw_folder["id"]))
                    trashed = True
                record = {
                    "type": "raw_folder",
                    "raw_folder_id": raw_folder.get("id"),
                    "raw_folder_name": raw_name,
                    "channel_code": raw_channel,
                    "expected_count": expected_count,
                    "generated_found_count": generated_found,
                    "labeled_found_count": labeled_found,
                    "generated_ratio": round(generated_ratio, 4),
                    "labeled_ratio": round(labeled_ratio, 4),
                    "decision": decision,
                    "trashed": trashed,
                    "matched_folder_ids": sorted({str(match["folder_id"]) for match in matched}),
                    "matched_buckets": sorted({str(match["bucket"]) for match in matched}),
                }
            processed += 1
            counts[record["decision"]] += 1
            out.write(json.dumps(record, sort_keys=True) + "\n")

        footer = {
            "type": "summary_footer",
            "processed_raw_folders": processed,
            "decision_counts": dict(sorted(counts.items())),
        }
        out.write(json.dumps(footer, sort_keys=True) + "\n")

    print(json.dumps({"output": str(output_path), **footer}, sort_keys=True))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="restaurant-pi-1", help="Reolink site key to audit.")
    parser.add_argument("--channel", help="Optional channel filter, e.g. CH05, 5, IPC5.")
    parser.add_argument("--limit", type=int, help="Maximum raw true-10 folders to audit.")
    parser.add_argument("--min-generated-ratio", type=float, default=0.80, help="Eligible threshold. Default: 0.80.")
    parser.add_argument("--require-labeled", action="store_true", help="Require labeled artifacts to meet the threshold.")
    parser.add_argument("--mode", choices=("dry-run", "trash"), default="dry-run")
    parser.add_argument("--confirm", default="", help=f"Required for --mode trash: {CONFIRM_TEXT}")
    parser.add_argument("--output", help="JSONL report path. Default: /private/tmp/screenrecord_true_ten_reconcile_<timestamp>.jsonl")
    args = parser.parse_args(argv)
    if not 0 < args.min_generated_ratio <= 1:
        parser.error("--min-generated-ratio must be > 0 and <= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


if __name__ == "__main__":
    raise SystemExit(reconcile(parse_args(sys.argv[1:])))
