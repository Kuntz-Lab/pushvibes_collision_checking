#!/usr/bin/env python3
"""Extract obstacle PC and RGB image from rosbags (one frame each)."""

import struct
import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, Stores

store = get_typestore(Stores.ROS1_NOETIC)

DATA_DIR = Path("data/PushVIB3S_procedure_level_plan_example")
BAGS = {
    1: DATA_DIR / "rosbags/thanks_joe_v1_2026-05-29-18-20-56.bag",
    2: DATA_DIR / "rosbags/thanks_joe_v2_2026-05-29-18-23-28.bag",
    3: DATA_DIR / "rosbags/thanks_joe_v3_2026-05-29-18-26-06.bag",
}

OBSTACLE_TOPIC = "/obstacle_pointcloud"
TISSUE_TOPIC = "/tissue_pointcloud"
IMAGE_TOPIC = "/camera/color/image_raw"


def read_pointcloud2(msg) -> np.ndarray:
    """Parse a PointCloud2 message into an (N, 3) float32 XYZ array."""
    fields = {f.name: f for f in msg.fields}
    x_off = fields["x"].offset
    y_off = fields["y"].offset
    z_off = fields["z"].offset
    point_step = msg.point_step
    data = msg.data
    n = msg.width * msg.height
    pts = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        base = i * point_step
        pts[i, 0] = struct.unpack_from("f", data, base + x_off)[0]
        pts[i, 1] = struct.unpack_from("f", data, base + y_off)[0]
        pts[i, 2] = struct.unpack_from("f", data, base + z_off)[0]
    # drop NaN/inf points
    valid = np.isfinite(pts).all(axis=1)
    return pts[valid]


def read_image(msg) -> np.ndarray:
    """Parse a sensor_msgs/Image into an HxWxC uint8 array."""
    h, w = msg.height, msg.width
    raw = bytes(msg.data)
    if msg.encoding in ("rgb8", "bgr8"):
        img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
    elif msg.encoding == "mono8":
        img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    return img


def first_message(bag, topic):
    """Return the first deserialized message for a topic."""
    conns = [c for c in bag.connections if c.topic == topic]
    if not conns:
        return None
    for conn, ts, rawdata in bag.messages(connections=conns):
        return store.deserialize_ros1(rawdata, conn.msgtype)
    return None


for v, bag_path in BAGS.items():
    print(f"\n{'='*60}")
    print(f"Bag v{v}: {bag_path.name}")
    print("=" * 60)
    with Reader(bag_path) as bag:
        obstacle_msg = first_message(bag, OBSTACLE_TOPIC)
        tissue_msg = first_message(bag, TISSUE_TOPIC)
        image_msg = first_message(bag, IMAGE_TOPIC)

    if obstacle_msg is not None:
        pc = read_pointcloud2(obstacle_msg)
        out = DATA_DIR / f"obstacle_pc_v{v}.npy"
        np.save(out, pc)
        print(f"Obstacle PC: {pc.shape}  saved -> {out.name}")
    else:
        print("Obstacle PC: topic not found")

    if tissue_msg is not None:
        pc = read_pointcloud2(tissue_msg)
        out = DATA_DIR / f"tissue_pc_v{v}.npy"
        np.save(out, pc)
        print(f"Tissue PC:   {pc.shape}  saved -> {out.name}")
    else:
        print("Tissue PC: topic not found")

    if image_msg is not None:
        img = read_image(image_msg)
        out = DATA_DIR / f"image_v{v}.npy"
        np.save(out, img)
        print(f"Image:       {img.shape} dtype={img.dtype}  saved -> {out.name}")
        # also save as PNG for quick visual inspection
        try:
            import PIL.Image
            png_out = DATA_DIR / f"image_v{v}.png"
            PIL.Image.fromarray(img).save(png_out)
            print(f"             also saved -> {png_out.name}")
        except ImportError:
            print("             (install Pillow to save PNG)")
    else:
        print("Image: topic not found")
