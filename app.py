#!/usr/bin/env python3
"""
Video Table Cropper - Flask Web App

Drag-and-drop interface to crop videos based on JSON bounding boxes.
Supports both axis-aligned and rotated bounding boxes.
"""

import io
import json
import math
import os
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
OUTPUT_FOLDER = Path(__file__).parent / "output"
FRAMES_FOLDER = Path(__file__).parent / "frames"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_JSON_EXTENSIONS = {".json"}

# Ensure folders exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)
FRAMES_FOLDER.mkdir(exist_ok=True)


# Camera JSON configurations (IPC3 through IPC11)
CAMERA_CONFIGS = {
    "IPC3": {
        "video_name": "3_1_Mimosas_IPC3_20251227103309",
        "frame_width": 1280,
        "frame_height": 720,
        "tables": [
            {"id": 0, "rotated_bbox": {"center": [822.78662109375, 332.6180419921875], "size": [321.945618340476, 526.1953508124054], "angle": -67.11446052236897, "corners": [[1002.5732421875, 583.236083984375], [517.798095703125, 378.60321044921875], [643.0, 82.0], [1127.775146484375, 286.63287353515625]]}, "saved": True},
            {"id": 1, "rotated_bbox": {"center": [282.5, 314.5], "size": [178.8627565551128, 206.3446493551661], "angle": 47.52611691161956, "corners": [[146.0126953125, 318.20428466796875], [298.20947265625, 178.86920166015625], [418.9873046875, 310.79571533203125], [266.79052734375, 450.13079833984375]]}, "saved": True},
            {"id": 2, "rotated_bbox": {"center": [563.73388671875, 609.4105224609375], "size": [477.8999734397775, 298.649962841451], "angle": 31.06887599786191, "corners": [[282.0, 614.0], [436.123779296875, 358.192138671875], [845.4677734375, 604.821044921875], [691.343994140625, 860.62890625]]}, "saved": True},
            {"id": 3, "rotated_bbox": {"center": [338.9871520996094, 105.39764404296875], "size": [142.12557867105397, 180.8465860855565], "angle": 34.84801602923692, "corners": [[229.0, 139.0], [332.33599853515625, -9.415489196777344], [448.97430419921875, 71.7952880859375], [345.6383056640625, 220.21078491210938]]}, "saved": True},
            {"id": 4, "rotated_bbox": {"center": [144.28004455566406, 237.20889282226562], "size": [122.63448438035192, 169.83926620137368], "angle": 52.96, "corners": [[39.56009292602539, 239.4178009033203], [175.12835693359375, 137.11129760742188], [249.0, 234.99998474121094], [113.43173217773438, 337.3064880371094]]}, "saved": True},
            {"id": 5, "rotated_bbox": {"center": [952.7930297851562, 161.791015625], "size": [255.25362446914727, 90.62169592974016], "angle": 19.76, "corners": [[817.362548828125, 161.28562927246094], [848.0, 76.00001525878906], [1088.2235107421875, 162.29640197753906], [1057.5860595703125, 247.58201599121094]]}, "saved": True},
            {"id": 6, "rotated_bbox": {"center": [508.89788818359375, 159.76670837402344], "size": [196.48954465504943, 167.32461181541007], "angle": 21.423695831610086, "corners": [[386.8826904296875, 201.76327514648438], [448.0, 45.999996185302734], [630.9130859375, 117.7701416015625], [569.7957763671875, 273.5334167480469]]}, "saved": True},
            {"id": 7, "rotated_bbox": {"center": [228.309814453125, 94.77656555175781], "size": [88.27931189306211, 110.59560807619341], "angle": 39.80557109226519, "corners": [[159.0, 108.99998474121094], [229.80157470703125, 24.038097381591797], [297.61962890625, 80.55314636230469], [226.81805419921875, 165.51502990722656]]}, "saved": True},
            {"id": 8, "rotated_bbox": {"center": [82.22715759277344, 196.13192749023438], "size": [71.23191353350012, 150.0705930656038], "angle": 48.43766499236521, "corners": [[2.4543190002441406, 219.26385498046875], [114.7422866821289, 119.70184326171875], [162.0, 173.0], [49.71202850341797, 272.56201171875]]}, "saved": True},
            {"id": 9, "rotated_bbox": {"center": [36.64238739013672, 180.69140625], "size": [50.303291239986194, 112.0168003117322], "angle": 51.00900595749453, "corners": [[-22.715225219726562, 196.3828125], [64.34925842285156, 125.90203094482422], [96.0, 165.0], [8.935516357421875, 235.48077392578125]]}, "saved": True},
            {"id": 10, "rotated_bbox": {"center": [174.4415283203125, 76.7829818725586], "size": [80.17810262375383, 83.17323186369374], "angle": 21.26337159910141, "corners": [[121.99998474121094, 101.0], [152.16322326660156, 23.488929748535156], [226.88307189941406, 52.56596374511719], [196.71983337402344, 130.0770263671875]]}, "saved": True}
        ]
    },
    "IPC4": {
        "video_name": "4_1_Mimosas_IPC4_20251227103312",
        "frame_width": 1280,
        "frame_height": 720,
        "tables": [
            {"id": 0, "rotated_bbox": {"center": [580.5, 575.0], "size": [284.0, 413.0], "angle": 90.0, "corners": [[374.0, 433.0], [787.0, 433.0], [787.0, 717.0], [374.0, 717.0]]}, "saved": True},
            {"id": 1, "rotated_bbox": {"center": [509.28826904296875, 309.04315185546875], "size": [232.43705362085305, 324.5949692069938], "angle": 79.45525328601899, "corners": [[328.4632568359375, 224.48828125], [647.5765380859375, 165.0863037109375], [690.11328125, 393.5980224609375], [371.0, 453.0]]}, "saved": True},
            {"id": 2, "rotated_bbox": {"center": [1021.257568359375, 182.017822265625], "size": [177.15739315789966, 135.70969158094044], "angle": 22.38, "corners": [[913.51513671875, 211.03564453125], [965.186279296875, 85.5477523803711], [1129.0, 153.0], [1077.328857421875, 278.4878845214844]]}, "saved": True},
            {"id": 3, "rotated_bbox": {"center": [778.70166015625, 109.91230010986328], "size": [169.22262145461497, 130.1695092119923], "angle": 33.690067525979785, "corners": [[672.1982421875, 117.13217163085938], [744.4033203125, 8.824600219726562], [885.205078125, 102.69242858886719], [813.0, 211.0]]}, "saved": True},
            {"id": 4, "rotated_bbox": {"center": [193.35546875, 367.37518310546875], "size": [220.78610693586702, 283.46312072196145], "angle": 70.35, "corners": [[22.75546646118164, 311.071533203125], [289.7109375, 215.7503662109375], [363.9554748535156, 423.6788330078125], [97.0, 519.0]]}, "saved": True},
            {"id": 5, "rotated_bbox": {"center": [427.34100341796875, 187.24942016601562], "size": [113.42567776555816, 191.04182231184123], "angle": 88.83, "corners": [[330.6819763183594, 132.49884033203125], [521.6839599609375, 128.59796142578125], [524.0, 242.0], [332.998046875, 245.90087890625]]}, "saved": True},
            {"id": 6, "rotated_bbox": {"center": [862.3423461914062, 334.21337890625], "size": [261.51689776363946, 216.42461367169057], "angle": 36.53, "corners": [[692.859375, 343.333740234375], [821.6846923828125, 169.4267578125], [1031.8253173828125, 325.093017578125], [903.0, 499.0]]}, "saved": True}
        ]
    },
    "IPC5": {
        "video_name": "5_1_Mimosas_IPC5_20251227103314",
        "frame_width": 1280,
        "frame_height": 720,
        "tables": [
            {"id": 0, "rotated_bbox": {"center": [1078.795654296875, 529.7513427734375], "size": [253.62942923372705, 256.25924722345667], "angle": 34.02, "corners": [[902.0, 565.0000610351562], [1045.37255859375, 352.60150146484375], [1255.59130859375, 494.50262451171875], [1112.21875, 706.9011840820312]]}, "saved": True},
            {"id": 1, "rotated_bbox": {"center": [157.16436767578125, 355.8930358886719], "size": [195.05875920649686, 215.52394181184565], "angle": 61.78, "corners": [[16.093563079833984, 320.91229248046875], [206.0, 219.0], [298.23516845703125, 390.873779296875], [108.3287353515625, 492.78607177734375]]}, "saved": True},
            {"id": 2, "rotated_bbox": {"center": [538.77783203125, 185.7633819580078], "size": [150.85503780427098, 280.8961493672101], "angle": 69.54, "corners": [[380.8237609863281, 164.18820190429688], [644.0, 65.99999237060547], [696.73193359375, 207.33856201171875], [433.5556640625, 305.5267639160156]]}, "saved": True},
            {"id": 3, "rotated_bbox": {"center": [705.08642578125, 317.0765380859375], "size": [225.39132509474166, 577.0084354202078], "angle": 66.88117384157762, "corners": [[395.50250244140625, 326.70928955078125], [926.1728515625, 100.153076171875], [1014.6703491210938, 307.44378662109375], [484.0, 534.0]]}, "saved": True}
        ]
    },
    "IPC6": {
        "video_name": "6_1_Mimosas_IPC6_20251227103316",
        "frame_width": 1280,
        "frame_height": 720,
        "tables": [
            {"id": 0, "rotated_bbox": {"center": [522.4129028320312, 498.6118469238281], "size": [417.3054157696059, 359.3356143029141], "angle": 1.59, "corners": [[308.855224609375, 672.4208984375], [318.8258056640625, 313.2237243652344], [735.9705810546875, 324.80279541015625], [726.0, 684.0]]}, "saved": True},
            {"id": 1, "rotated_bbox": {"center": [1088.203369140625, 175.82452392578125], "size": [89.01743313801416, 115.44641728423342], "angle": 75.96, "corners": [[1021.4067993164062, 146.64903259277344], [1133.4044189453125, 118.64183044433594], [1155.0, 205.00001525878906], [1043.0023193359375, 233.00721740722656]]}, "saved": True},
            {"id": 2, "rotated_bbox": {"center": [1048.9993896484375, 295.5008850097656], "size": [148.36497910017403, 73.86854117134503], "angle": 38.66, "corners": [[968.0, 278.0], [1014.1455078125, 220.3185272216797], [1129.998779296875, 313.00177001953125], [1083.853271484375, 370.6832275390625]]}, "saved": True},
            {"id": 3, "rotated_bbox": {"center": [155.53573608398438, 321.8195495605469], "size": [209.64128669457634, 216.4568417538395], "angle": 70.29, "corners": [[18.296466827392578, 259.6412353515625], [222.0714874267578, 186.63909912109375], [292.7749938964844, 383.99786376953125], [88.99998474121094, 457.0]]}, "saved": True},
            {"id": 4, "rotated_bbox": {"center": [950.0, 155.0], "size": [129.05849282081135, 89.01632114854323], "angle": 32.11, "corners": [[871.6837158203125, 158.3994140625], [919.0, 83.0], [1028.3162841796875, 151.6005859375], [981.0, 227.0]]}, "saved": True},
            {"id": 5, "rotated_bbox": {"center": [1106.6502685546875, 245.05596923828125], "size": [121.43250399275837, 61.66508206721707], "angle": 18.43, "corners": [[1039.300537109375, 255.1119384765625], [1058.795654296875, 196.609619140625], [1174.0, 235.0], [1154.5048828125, 293.5023193359375]]}, "saved": True},
            {"id": 6, "rotated_bbox": {"center": [841.7884521484375, 398.2997131347656], "size": [193.89737469280215, 280.46298822926275], "angle": 71.45, "corners": [[678.0000610351562, 351.0], [943.8919067382812, 261.77569580078125], [1005.5768432617188, 445.59942626953125], [739.6849975585938, 534.82373046875]]}, "saved": True},
            {"id": 7, "rotated_bbox": {"center": [635.0, 186.0], "size": [187.02165892190968, 180.6623898161897], "angle": 44.89054524803136, "corners": [[505.0, 184.0], [632.5032958984375, 56.00859069824219], [765.0, 188.0], [637.4967041015625, 315.99139404296875]]}, "saved": True},
            {"id": 8, "rotated_bbox": {"center": [820.6773071289062, 159.86863708496094], "size": [189.10743406121054, 123.64285988886165], "angle": 29.05, "corners": [[708.0, 168.0], [768.03759765625, 59.911869049072266], [933.3546142578125, 151.73727416992188], [873.3170166015625, 259.8254089355469]]}, "saved": True},
            {"id": 9, "rotated_bbox": {"center": [386.4109802246094, 227.96266174316406], "size": [183.14332984146836, 200.3973217727612], "angle": 65.14, "corners": [[257.0, 187.0], [438.82806396484375, 102.75247192382812], [515.8219604492188, 268.9253234863281], [333.993896484375, 353.1728515625]]}, "saved": True}
        ]
    },
    "IPC7": {
        "video_name": "7_1_Mimosas_IPC7_20251227112204",
        "frame_width": 1280,
        "frame_height": 720,
        "tables": [
            {"id": 0, "rotated_bbox": {"center": [535.4974975585938, 451.677978515625], "size": [511.7046029142641, 294.49656205365386], "angle": 52.58528303002687, "corners": [[263.09356689453125, 337.93017578125], [496.9999694824219, 159.0], [807.9014282226562, 565.42578125], [573.9949951171875, 744.35595703125]]}, "saved": True},
            {"id": 1, "rotated_bbox": {"center": [1135.403076171875, 169.1923828125], "size": [195.87468534201227, 71.55471642984267], "angle": 26.57, "corners": [[1031.80615234375, 157.384765625], [1063.81201171875, 93.38705444335938], [1239.0, 181.0], [1206.994140625, 244.99771118164062]]}, "saved": True},
            {"id": 2, "rotated_bbox": {"center": [135.19793701171875, 549.2426147460938], "size": [438.18117132951545, 334.05306338160386], "angle": 74.05, "corners": [[-85.6041259765625, 384.4852294921875], [235.58851623535156, 292.68798828125], [356.0, 714.0], [34.80735778808594, 805.7972412109375]]}, "saved": True},
            {"id": 3, "rotated_bbox": {"center": [1071.939697265625, 198.31060791015625], "size": [236.80291594594402, 68.18404942363028], "angle": 29.05, "corners": [[951.87939453125, 170.6212158203125], [984.98779296875, 111.01498413085938], [1192.0, 226.0], [1158.8916015625, 285.6062316894531]]}, "saved": True},
            {"id": 4, "rotated_bbox": {"center": [819.973388671875, 310.96209716796875], "size": [392.24096577367425, 223.21045643025758], "angle": 29.760559189317814, "corners": [[594.321533203125, 310.497802734375], [705.117919921875, 116.72718811035156], [1045.625244140625, 311.4263916015625], [934.828857421875, 505.197021484375]]}, "saved": True},
            {"id": 5, "rotated_bbox": {"center": [969.3091430664062, 227.88922119140625], "size": [312.6041263371481, 98.83466386475003], "angle": 26.57, "corners": [[807.410400390625, 202.175048828125], [851.6182861328125, 113.77843475341797], [1131.2078857421875, 253.6033935546875], [1087.0, 342.0]]}, "saved": True},
            {"id": 6, "rotated_bbox": {"center": [190.27, 704.49], "size": [103.21, 38.42], "angle": 20.06, "corners": [[135.20655822753906, 704.8338623046875], [148.38478088378906, 668.74462890625], [245.3334503173828, 704.1461181640625], [232.1552276611328, 740.2353515625]]}, "saved": True}
        ]
    },
    "IPC8": {
        "video_name": "8_1_Mimosas_IPC8_20251227103321",
        "frame_width": 1280,
        "frame_height": 720,
        "tables": [
            {"id": 0, "rotated_bbox": {"center": [424.5, 546.0], "size": [565.4548287426129, 305.086605165914], "angle": 24.421943574652165, "corners": [[104.0, 568.0], [230.1390380859375, 290.2109069824219], [745.0, 524.0], [618.8609619140625, 801.7890625]]}, "saved": True}
        ]
    },
    "IPC9": {
        "video_name": "9_1_Mimosas_IPC9_20251227103325",
        "frame_width": 1280,
        "frame_height": 720,
        "tables": [
            {"id": 0, "rotated_bbox": {"center": [654.2850341796875, 204.67372131347656], "size": [153.04692549671307, 455.3847732184601], "angle": 54.89760902044744, "corners": [[424.0000305175781, 273.0], [796.56201171875, 11.135826110839844], [884.570068359375, 136.34744262695312], [512.008056640625, 398.21160888671875]]}, "saved": True}
        ]
    },
    "IPC10": {
        "video_name": "10_1_Mimosas_IPC10_20251227112207",
        "frame_width": 1280,
        "frame_height": 720,
        "tables": [
            {"id": 0, "rotated_bbox": {"center": [408.0, 386.5], "size": [255.12070966688265, 610.1658983416446], "angle": 34.844820406223185, "corners": [[129.0, 564.0], [477.621826171875, 63.235313415527344], [687.0, 209.0], [338.378173828125, 709.7647094726562]]}, "saved": True}
        ]
    },
    "IPC11": {
        "video_name": "11_1_Mimosas_IPC11_20251227103501",
        "frame_width": 1280,
        "frame_height": 720,
        "tables": [
            {"id": 0, "rotated_bbox": {"center": [539.0, 268.0], "size": [597.2972663831904, 148.0809763986184], "angle": 32.3, "corners": [[247.0, 170.99998474121094], [326.12744140625, 45.832794189453125], [831.0, 365.0], [751.87255859375, 490.1672058105469]]}, "saved": True}
        ]
    }
}


def detect_camera_from_filename(filename: str) -> str:
    """Extract camera ID from filename (e.g., 'IPC3' from '3_1_Mimosas_IPC3_...')."""
    import re
    match = re.search(r'IPC(\d+)', filename, re.IGNORECASE)
    if match:
        return f"IPC{match.group(1)}"
    return None


def get_camera_config(camera_id: str) -> dict:
    """Get stored JSON config for a camera."""
    return CAMERA_CONFIGS.get(camera_id)


def allowed_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_json(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_JSON_EXTENSIONS


def crop_video(input_path: Path, output_path: Path, x1: int, y1: int, x2: int, y2: int) -> bool:
    """Crop a video using ffmpeg (axis-aligned bbox)."""
    width = x2 - x1
    height = y2 - y1
    crop_filter = f"crop={width}:{height}:{x1}:{y1}"

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", crop_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        "-y",
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def crop_rotated_video(input_path: Path, output_path: Path, rotated_bbox: dict, frame_width: int, frame_height: int) -> bool:
    """
    Crop a rotated rectangular region from video.

    Strategy:
    1. Crop the axis-aligned bounding box containing all corners
    2. Pad to center the rotation point
    3. Rotate to straighten the region
    4. Crop the final rectangle from center
    """
    center = rotated_bbox.get("center", [0, 0])
    size = rotated_bbox.get("size", [100, 100])
    angle = rotated_bbox.get("angle", 0)
    corners = rotated_bbox.get("corners", [])

    if not corners or len(corners) < 4:
        return False

    cx, cy = center
    final_w, final_h = int(round(size[0])), int(round(size[1]))

    # Make dimensions even for codec compatibility
    final_w = final_w + (final_w % 2)
    final_h = final_h + (final_h % 2)

    # Get axis-aligned bounding box of corners
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]

    margin = 10
    min_x = max(0, int(min(xs)) - margin)
    min_y = max(0, int(min(ys)) - margin)
    max_x = min(frame_width, int(max(xs)) + margin + 1)
    max_y = min(frame_height, int(max(ys)) + margin + 1)

    # First crop dimensions
    crop1_w = max_x - min_x
    crop1_h = max_y - min_y
    crop1_x = min_x
    crop1_y = min_y

    # Bbox center in cropped coordinates
    new_cx = cx - min_x
    new_cy = cy - min_y

    # Calculate padded size (large enough to center bbox and fit final crop after rotation)
    S = 2 * int(max(new_cx, new_cy, crop1_w - new_cx, crop1_h - new_cy)) + max(final_w, final_h) + 100
    S = S + (S % 2)  # Make even

    # Padding to center bbox at (S/2, S/2)
    pad_left = int(S / 2 - new_cx)
    pad_top = int(S / 2 - new_cy)
    pad_right = S - crop1_w - pad_left
    pad_bottom = S - crop1_h - pad_top

    # Ensure non-negative padding
    if pad_left < 0:
        pad_right -= pad_left
        pad_left = 0
    if pad_top < 0:
        pad_bottom -= pad_top
        pad_top = 0
    if pad_right < 0:
        pad_left -= pad_right
        pad_right = 0
    if pad_bottom < 0:
        pad_top -= pad_bottom
        pad_bottom = 0

    # Recalculate S after adjustments
    S = crop1_w + pad_left + pad_right
    S_h = crop1_h + pad_top + pad_bottom
    if S != S_h:
        # Make square by adding to the smaller dimension
        if S > S_h:
            diff = S - S_h
            pad_bottom += diff
        else:
            diff = S_h - S
            pad_right += diff
            S = S_h

    # Rotation angle (negative to un-rotate)
    angle_rad = -angle * math.pi / 180

    # Final crop position (centered)
    final_crop_x = int((S - final_w) / 2)
    final_crop_y = int((S - final_h) / 2)

    # Build ffmpeg filter chain
    filter_str = (
        f"crop={crop1_w}:{crop1_h}:{crop1_x}:{crop1_y},"
        f"pad={S}:{S}:{pad_left}:{pad_top}:black,"
        f"rotate={angle_rad}:ow={S}:oh={S}:c=black,"
        f"crop={final_w}:{final_h}:{final_crop_x}:{final_crop_y}"
    )

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", filter_str,
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        "-y",
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_video_info(video_path: Path) -> dict:
    """Get video information using ffprobe."""
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,r_frame_rate,width,height",
        "-of", "json",
        str(video_path)
    ]

    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}

        data = json.load(io.StringIO(result.stdout))
        if not data.get("streams"):
            return {}

        stream = data["streams"][0]

        # Parse frame rate
        fps_str = stream.get("r_frame_rate", "30/1")
        fps_parts = fps_str.split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1])

        return {
            "duration": float(stream.get("duration", 0)),
            "fps": fps,
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "resolution": f"{stream.get('width')}x{stream.get('height')}"
        }

    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return {}


def format_timestamp(seconds: float) -> str:
    """Convert seconds to timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
    else:
        return f"{minutes:02d}m{secs:02d}s"


def extract_frames_from_video(video_path: Path, output_dir: Path, interval: int = 30, quality: int = 2) -> bool:
    """Extract frames from video at regular intervals."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video info
    video_info = get_video_info(video_path)
    if not video_info:
        return False

    # Extract frames using ffmpeg
    temp_pattern = str(output_dir / "temp_%04d.jpg")
    fps_value = f"1/{interval}"

    extract_cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps_value}",
        "-q:v", str(quality),
        "-y",
        temp_pattern
    ]

    try:
        result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return False

    except subprocess.TimeoutExpired:
        return False

    # Rename frames to include timestamps
    temp_frames = sorted(output_dir.glob("temp_*.jpg"))
    frames_info = []

    for idx, temp_frame in enumerate(temp_frames):
        timestamp_sec = idx * interval
        timestamp_str = format_timestamp(timestamp_sec)

        new_name = f"frame_{idx:04d}_{timestamp_str}.jpg"
        new_path = output_dir / new_name

        temp_frame.rename(new_path)

        frames_info.append({
            "frame_number": idx,
            "filename": new_name,
            "timestamp_seconds": timestamp_sec,
            "timestamp_formatted": format_timestamp(timestamp_sec),
            "file_size_bytes": new_path.stat().st_size
        })

    # Generate metadata
    from datetime import datetime
    metadata = {
        "video_info": {
            "source_video": str(video_path.name),
            "video_name": video_path.stem,
            "duration_seconds": video_info.get("duration", 0),
            "fps": video_info.get("fps", 0),
            "resolution": video_info.get("resolution", "unknown"),
            "extraction_date": datetime.now().isoformat()
        },
        "extraction_params": {
            "interval_seconds": interval,
            "format": "jpg",
            "quality": quality,
            "total_frames_extracted": len(frames_info)
        },
        "frames": frames_info
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/frames")
def frames_page():
    return render_template("frames.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle file uploads."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename
    file_type = request.form.get("type", "unknown")

    # Generate unique filename to avoid collisions
    unique_id = str(uuid.uuid4())[:8]
    ext = Path(filename).suffix
    safe_name = f"{unique_id}_{Path(filename).stem}{ext}"
    save_path = UPLOAD_FOLDER / safe_name

    if file_type == "video" and allowed_video(filename):
        file.save(save_path)
        return jsonify({"success": True, "filename": safe_name, "original": filename})
    elif file_type == "json" and allowed_json(filename):
        file.save(save_path)
        # Parse and return table info
        try:
            with open(save_path) as f:
                data = json.load(f)
            tables = data.get("tables", [])
            return jsonify({
                "success": True,
                "filename": safe_name,
                "original": filename,
                "tables": len(tables),
                "video_name": data.get("video_name", "unknown")
            })
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON file"}), 400
    else:
        return jsonify({"error": f"Invalid file type: {filename}"}), 400


@app.route("/cameras", methods=["GET"])
def get_cameras():
    """Return list of available cameras."""
    cameras = []
    for camera_id, config in CAMERA_CONFIGS.items():
        cameras.append({
            "id": camera_id,
            "name": f"Camera {camera_id.replace('IPC', '')}",
            "tables": len(config.get("tables", []))
        })
    return jsonify({"cameras": sorted(cameras, key=lambda x: x["id"])})


@app.route("/process", methods=["POST"])
def process():
    """Process multiple videos with their assigned cameras."""
    data = request.json
    videos = data.get("videos", [])  # Array of {filename, camera}

    if not videos:
        return jsonify({"error": "No videos provided"}), 400

    # Validate all videos exist and have camera assignments
    for video in videos:
        if not video.get("filename") or not video.get("camera"):
            return jsonify({"error": "Each video needs filename and camera"}), 400

        video_path = UPLOAD_FOLDER / video["filename"]
        if not video_path.exists():
            return jsonify({"error": f"Video not found: {video['filename']}"}), 404

        camera_config = get_camera_config(video["camera"])
        if not camera_config:
            return jsonify({"error": f"Unknown camera: {video['camera']}"}), 400

    # Create unique output folder for this batch job
    job_id = str(uuid.uuid4())[:8]
    job_output = OUTPUT_FOLDER / job_id
    job_output.mkdir(exist_ok=True)

    results = []

    # Process each video with its camera config
    for video in videos:
        video_filename = video["filename"]
        camera_id = video["camera"]
        video_path = UPLOAD_FOLDER / video_filename
        camera_config = get_camera_config(camera_id)

        tables = camera_config.get("tables", [])
        frame_width = camera_config.get("frame_width", 1280)
        frame_height = camera_config.get("frame_height", 720)

        # Process each table in this camera's config
        for idx, table in enumerate(tables):
            if not table.get("saved", True) or table.get("skip_reason"):
                continue

            table_id = table.get("id", idx)
            # Include camera ID in output filename
            output_name = f"{camera_id}_{video_path.stem}_table_{table_id:02d}.mp4"
            output_path = job_output / output_name

            # Crop using rotated_bbox or bbox
            rotated_bbox = table.get("rotated_bbox")
            bbox = table.get("bbox", {})

            success = False
            bbox_info = {}

            if rotated_bbox and rotated_bbox.get("corners"):
                success = crop_rotated_video(
                    video_path, output_path, rotated_bbox, frame_width, frame_height
                )
                bbox_info = {
                    "center": rotated_bbox.get("center"),
                    "size": rotated_bbox.get("size"),
                    "angle": rotated_bbox.get("angle")
                }
            elif bbox:
                x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
                if x2 > x1 and y2 > y1:
                    success = crop_video(video_path, output_path, x1, y1, x2, y2)
                bbox_info = bbox

            if success:
                results.append({
                    "table_id": table_id,
                    "camera": camera_id,  # NEW: Track source camera
                    "filename": output_name,
                    "download_url": f"/download/{job_id}/{output_name}",
                    "bbox": bbox_info,
                    "video_path": output_path
                })

    # Auto-extract frames from all cropped videos
    labeling_job_id = str(uuid.uuid4())[:8]
    labeling_folder = FRAMES_FOLDER / labeling_job_id
    labeling_folder.mkdir(parents=True, exist_ok=True)

    frame_metadata = []
    for result in results:
        cropped_video_path = result["video_path"]
        video_frames_dir = labeling_folder / f"{cropped_video_path.stem}_frames"

        extract_success = extract_frames_from_video(
            cropped_video_path,
            video_frames_dir,
            interval=30,
            quality=2
        )

        if extract_success:
            for frame in sorted(video_frames_dir.glob("frame_*.jpg")):
                frame_metadata.append({
                    "filename": frame.name,
                    "table_video": cropped_video_path.stem,
                    "camera": result["camera"],  # NEW: Include camera source
                    "relative_path": f"{video_frames_dir.name}/{frame.name}"
                })

    # Save frame metadata
    metadata_file = labeling_folder / "frames_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(frame_metadata, f, indent=2)

    # Clean up video_path from results (not JSON serializable)
    for result in results:
        del result["video_path"]

    return jsonify({
        "success": True,
        "job_id": job_id,
        "videos": results,
        "count": len(results),
        "labeling_job_id": labeling_job_id,
        "frame_count": len(frame_metadata),
        "cameras_processed": list(set(v["camera"] for v in videos))  # NEW
    })


@app.route("/download/<job_id>/<filename>")
def download(job_id: str, filename: str):
    """Download a cropped video."""
    file_path = OUTPUT_FOLDER / job_id / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)


@app.route("/download-zip/<job_id>")
def download_zip(job_id: str):
    """Download all cropped videos as a ZIP file."""
    job_folder = OUTPUT_FOLDER / job_id
    if not job_folder.exists():
        return jsonify({"error": "Job not found"}), 404

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in job_folder.iterdir():
            if file_path.is_file():
                zf.write(file_path, file_path.name)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"cropped_videos_{job_id}.zip"
    )


@app.route("/cleanup", methods=["POST"])
def cleanup():
    """Clean up uploaded and output files."""
    try:
        # Clear uploads
        for f in UPLOAD_FOLDER.iterdir():
            if f.is_file():
                f.unlink()
        # Clear outputs
        for d in OUTPUT_FOLDER.iterdir():
            if d.is_dir():
                shutil.rmtree(d)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload-frame-video", methods=["POST"])
def upload_frame_video():
    """Handle video upload for frame extraction."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename

    if not allowed_video(filename):
        return jsonify({"error": f"Invalid file type: {filename}"}), 400

    # Generate unique filename
    unique_id = str(uuid.uuid4())[:8]
    ext = Path(filename).suffix
    safe_name = f"{unique_id}_{Path(filename).stem}{ext}"
    save_path = UPLOAD_FOLDER / safe_name

    file.save(save_path)

    # Get video info
    video_info = get_video_info(save_path)

    return jsonify({
        "success": True,
        "filename": safe_name,
        "original": filename,
        "duration": video_info.get("duration", 0),
        "resolution": video_info.get("resolution", "unknown"),
        "fps": video_info.get("fps", 0)
    })


@app.route("/process-frames", methods=["POST"])
def process_frames():
    """Process video and extract frames."""
    data = request.json
    video_filename = data.get("video")
    interval = int(data.get("interval", 30))
    quality = int(data.get("quality", 2))

    if not video_filename:
        return jsonify({"error": "Missing video file"}), 400

    video_path = UPLOAD_FOLDER / video_filename
    if not video_path.exists():
        return jsonify({"error": "Video file not found"}), 404

    # Create unique output folder for this job
    job_id = str(uuid.uuid4())[:8]
    job_output = FRAMES_FOLDER / job_id
    job_output.mkdir(parents=True, exist_ok=True)

    # Extract frames
    success = extract_frames_from_video(video_path, job_output, interval, quality)

    if not success:
        return jsonify({"error": "Frame extraction failed"}), 500

    # Count extracted frames
    frame_files = list(job_output.glob("frame_*.jpg"))

    # Read metadata
    metadata_path = job_output / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return jsonify({
        "success": True,
        "job_id": job_id,
        "frame_count": len(frame_files),
        "metadata": metadata,
        "download_url": f"/download-frames/{job_id}"
    })


@app.route("/download-frames/<job_id>")
def download_frames(job_id: str):
    """Download all extracted frames as a ZIP file."""
    job_folder = FRAMES_FOLDER / job_id
    if not job_folder.exists():
        return jsonify({"error": "Job not found"}), 404

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in job_folder.iterdir():
            if file_path.is_file():
                zf.write(file_path, file_path.name)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"frames_{job_id}.zip"
    )


@app.route("/label/<job_id>")
def label_page(job_id: str):
    """Serve labeling interface for a frame extraction job."""
    return render_template("labeling.html", job_id=job_id)


@app.route("/get-frames/<job_id>")
def get_frames(job_id: str):
    """Get list of frames for labeling."""
    job_folder = FRAMES_FOLDER / job_id
    if not job_folder.exists():
        return jsonify({"error": "Job not found"}), 404

    # Read metadata
    metadata_path = job_folder / "frames_metadata.json"
    frames = []

    if metadata_path.exists():
        with open(metadata_path) as f:
            frames = json.load(f)
    else:
        # Fallback: scan directory
        for frame_file in job_folder.rglob("frame_*.jpg"):
            frames.append({
                "filename": frame_file.name,
                "table_video": "unknown",
                "relative_path": str(frame_file.relative_to(job_folder))
            })

    return jsonify({
        "success": True,
        "frames": frames,
        "frame_count": len(frames)
    })


@app.route("/frame-image/<job_id>/<path:filename>")
def serve_frame(job_id: str, filename: str):
    """Serve individual frame image."""
    # Handle nested paths (video_frames/frame_0001.jpg)
    frame_path = FRAMES_FOLDER / job_id / filename
    if not frame_path.exists():
        return jsonify({"error": "Frame not found"}), 404

    return send_file(frame_path, mimetype="image/jpeg")


@app.route("/download-labeled/<job_id>", methods=["POST"])
def download_labeled(job_id: str):
    """Download labeled frames organized by category."""
    data = request.json
    labels = data.get("labels", {})  # {relative_path: "clean"|"occupied"|"dirty"}

    job_folder = FRAMES_FOLDER / job_id
    if not job_folder.exists():
        return jsonify({"error": "Job not found"}), 404

    # Create ZIP with folders
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add frames to appropriate folders
        for relative_path, label in labels.items():
            # relative_path is like "table_00_frames/frame_0001_00m30s.jpg"
            frame_path = job_folder / relative_path

            if frame_path.exists():
                # Create a unique name: table_name + filename
                # e.g., "table_00_frame_0001_00m30s.jpg"
                parts = relative_path.split('/')
                if len(parts) >= 2:
                    folder_name = parts[0].replace('_frames', '')
                    file_name = parts[-1]
                    unique_name = f"{folder_name}_{file_name}"
                else:
                    unique_name = frame_path.name

                # Add to ZIP in category folder
                zf.write(frame_path, f"{label}/{unique_name}")

        # Add metadata
        metadata = {
            "labels": labels,
            "counts": {
                "clean": sum(1 for l in labels.values() if l == "clean"),
                "occupied": sum(1 for l in labels.values() if l == "occupied"),
                "dirty": sum(1 for l in labels.values() if l == "dirty")
            },
            "total": len(labels)
        }
        zf.writestr("labels.json", json.dumps(metadata, indent=2))

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"labeled_frames_{job_id}.zip"
    )


if __name__ == "__main__":
    print("=" * 50)
    print("Video Table Cropper")
    print("=" * 50)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print()
    print("Open http://localhost:8080 in your browser")
    print("=" * 50)
    app.run(debug=True, port=8080)
