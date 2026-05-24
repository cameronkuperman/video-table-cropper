#!/usr/bin/env python3
"""Seed a tiny Azure Blob/Azurite dataset for local AutoLabeler smoke tests."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from blob_client import AzureBlobClient  # noqa: E402
from queue_metadata import build_folder_app_properties  # noqa: E402


def _jpg(label: str, color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (320, 180), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 300, 160), outline=(255, 255, 255), width=3)
    draw.text((34, 76), label, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _ensure_path(client: AzureBlobClient, path: str) -> str:
    current = ""
    for part in path.strip("/").split("/"):
        current = client.ensure_subfolder(current, part)
    return current


def _seed_label_folder(client: AzureBlobClient, parent_id: str, folder_name: str) -> None:
    folder_id = client.ensure_subfolder(parent_id, folder_name)
    frames = {}
    for idx, color in ((0, (45, 92, 132)), (1, (76, 122, 68)), (2, (132, 84, 45))):
        uploaded = client.upload_bytes(
            _jpg(f"{folder_name} frame {idx}", color),
            folder_id,
            f"frame_{idx}.jpg",
            mime_type="image/jpeg",
        )
        frames[f"frame_{idx}"] = str(uploaded["id"])
    client.upsert_bytes(
        folder_id,
        "metadata.json",
        json.dumps({"seed": True, "folder": folder_name}).encode("utf-8"),
        mime_type="application/json",
    )
    client.update_file_metadata(folder_id, {"appProperties": build_folder_app_properties(frames)})


def main() -> None:
    os.environ.setdefault("STORAGE_BACKEND", "azure")
    os.environ.setdefault("AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true")
    os.environ.setdefault("AZURE_BLOB_CONTAINER", "autolabeler-dev")
    os.environ.setdefault("AZURE_PROJECT_ROOT_PREFIX", "project-root")

    client = AzureBlobClient()
    root = client.ensure_subfolder("", os.environ["AZURE_PROJECT_ROOT_PREFIX"].strip("/"))

    for name in ("raw_videos", "temp_processing", "unlabeled", "clean", "dirty", "occupied", "label_later", "discarded"):
        client.ensure_subfolder(root, name)

    unlabeled = client.ensure_subfolder(root, "unlabeled")
    _seed_label_folder(client, unlabeled, "mimosas-seed_table_1_t0001")

    for site in ("restaurant-pi-1", "reolink-matthews-01"):
        site_root = client.ensure_subfolder(root, site)
        unassociated = client.ensure_subfolder(site_root, "unassociated")
        client.ensure_subfolder(site_root, "unlabeled")
        client.ensure_subfolder(site_root, "processed_raw")
        three_frame = _ensure_path(client, f"{site_root}/3frame/unlabeled")
        true_ten = _ensure_path(client, f"{site_root}/10frametrue/seed-node")
        _seed_label_folder(client, three_frame, f"{site}-ready_table_1_t0001")

        raw_group = client.ensure_subfolder(unassociated, "Reolink-CH-CH03_t0001")
        for idx in range(10):
            client.upload_bytes(
                _jpg(f"{site} raw {idx}", (40 + idx * 12, 75, 120)),
                raw_group,
                f"frame_{idx}.jpg",
                mime_type="image/jpeg",
            )
        client.upsert_bytes(
            raw_group,
            "metadata.json",
            json.dumps({"frames_per_triplet": 10, "seed": True}).encode("utf-8"),
            mime_type="application/json",
        )
        client.ensure_subfolder(true_ten, "Reolink-CH-CH03_t0001")

        if site == "reolink-matthews-01":
            configs = client.ensure_subfolder(site_root, "crop_configs")
            client.upsert_bytes(
                configs,
                "CH-CH03.json",
                json.dumps(
                    {
                        "version": 1,
                        "site_key": site,
                        "channel_code": "CH-CH03",
                        "reference": {"width": 320, "height": 180},
                        "crops": [
                            {
                                "name": "table_top_1",
                                "polygon": [[20, 20], [300, 20], [300, 160], [20, 160]],
                            }
                        ],
                    },
                    indent=2,
                ).encode("utf-8"),
                mime_type="application/json",
            )

    print(
        "Seeded Azurite/Azure Blob dataset "
        f"container={client.container_name} root={client.root_prefix}"
    )


if __name__ == "__main__":
    main()
