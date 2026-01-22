import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import cv2


@dataclass
class ImageFrame:
    t_sec: float
    img_bgr: np.ndarray


@dataclass
class ImuSample:
    t_sec: float
    gyro_rad_s: np.ndarray  # (3,)
    accel_m_s2: Optional[np.ndarray] = None  # (3,) if available


def _to_sec(stamp) -> float:
    """
    Convert ROS2 builtin_interfaces/msg/Time or (sec, nanosec) like objects to float seconds.
    """
    # rosbags provides msg.header.stamp.sec / .nanosec
    sec = float(getattr(stamp, "sec", 0))
    nsec = float(getattr(stamp, "nanosec", 0))
    return sec + nsec * 1e-9


def _decode_sensor_msgs_image(msg) -> np.ndarray:
    """
    Decode ROS2 sensor_msgs/msg/Image to BGR uint8.
    Supports common encodings: rgb8, bgr8, mono8, bgra8, rgba8.
    """
    h = int(msg.height)
    w = int(msg.width)
    enc = str(msg.encoding).lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("bgr8",):
        img = data.reshape(h, w, 3)
        return img.copy()
    if enc in ("rgb8",):
        img = data.reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc in ("mono8",):
        img = data.reshape(h, w)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc in ("bgra8",):
        img = data.reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc in ("rgba8",):
        img = data.reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    raise ValueError(f"Unsupported Image encoding: {msg.encoding}")


def _decode_compressed_image(msg) -> np.ndarray:
    """
    Decode ROS2 sensor_msgs/msg/CompressedImage to BGR uint8.
    """
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed for CompressedImage")
    return img


def iter_ros2_bag_images(
    bag_path: str,
    image_topic: str,
    image_type: str = "raw",
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> Iterator[ImageFrame]:
    """
    Iterate images from a ROS2 bag using `rosbags`.

    bag_path: rosbag2 folder or .mcap
    image_type: 'raw' (sensor_msgs/msg/Image) or 'compressed' (sensor_msgs/msg/CompressedImage)
    """
    from rosbags.rosbag2 import Reader
    from rosbags.serde import deserialize_cdr

    stride = max(1, int(stride))
    count = 0
    kept = 0

    with Reader(bag_path) as reader:
        # Filter connections by topic
        conns = [c for c in reader.connections if c.topic == image_topic]
        if not conns:
            raise RuntimeError(f"No connections found for image_topic={image_topic} in bag: {bag_path}")

        for conn, t, raw in reader.messages(connections=conns):
            if max_frames is not None and kept >= int(max_frames):
                break
            if (count % stride) != 0:
                count += 1
                continue
            msg = deserialize_cdr(raw, conn.msgtype)
            # timestamp
            if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                t_sec = _to_sec(msg.header.stamp)
            else:
                # fallback: reader timestamp is nanoseconds since epoch for rosbag2
                t_sec = float(t) * 1e-9

            if image_type.lower() == "raw":
                img = _decode_sensor_msgs_image(msg)
            elif image_type.lower() == "compressed":
                img = _decode_compressed_image(msg)
            else:
                raise ValueError("image_type must be 'raw' or 'compressed'")

            yield ImageFrame(t_sec=t_sec, img_bgr=img)
            kept += 1
            count += 1


def iter_ros2_bag_imu(
    bag_path: str,
    imu_topic: str,
) -> Iterator[ImuSample]:
    """
    Iterate IMU samples from a ROS2 bag (sensor_msgs/msg/Imu) using `rosbags`.
    """
    from rosbags.rosbag2 import Reader
    from rosbags.serde import deserialize_cdr

    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == imu_topic]
        if not conns:
            raise RuntimeError(f"No connections found for imu_topic={imu_topic} in bag: {bag_path}")

        for conn, t, raw in reader.messages(connections=conns):
            msg = deserialize_cdr(raw, conn.msgtype)
            if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                t_sec = _to_sec(msg.header.stamp)
            else:
                t_sec = float(t) * 1e-9

            gyro = np.array(
                [float(msg.angular_velocity.x), float(msg.angular_velocity.y), float(msg.angular_velocity.z)],
                dtype=np.float64,
            )
            accel = None
            if hasattr(msg, "linear_acceleration"):
                accel = np.array(
                    [
                        float(msg.linear_acceleration.x),
                        float(msg.linear_acceleration.y),
                        float(msg.linear_acceleration.z),
                    ],
                    dtype=np.float64,
                )
            yield ImuSample(t_sec=t_sec, gyro_rad_s=gyro, accel_m_s2=accel)


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


