#!/usr/bin/env python3
import os
import argparse
import yaml

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_ply(path, points, colors=None):
    """
    Save point cloud to a simple ASCII PLY file.

    Args:
        path: output file path.
        points: (N, 3) float array.
        colors: (N, 3) uint8 array in BGR or RGB. If None, white is used.
    """
    points = points.reshape(-1, 3)
    if colors is None:
        colors = np.full_like(points, 255, dtype=np.uint8)
    else:
        colors = colors.reshape(-1, 3).astype(np.uint8)

    num_points = points.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[2])} {int(c[1])} {int(c[0])}\n")


def build_rectification_and_sgbm(
    cam0_intr_path,
    cam1_intr_path,
    stereo_extr_path,
):
    intr0 = load_yaml(cam0_intr_path)
    intr1 = load_yaml(cam1_intr_path)
    ext = load_yaml(stereo_extr_path)

    K0 = np.asarray(intr0["K"], dtype=np.float64)
    D0 = np.asarray(intr0["D"], dtype=np.float64).reshape(-1, 1)
    K1 = np.asarray(intr1["K"], dtype=np.float64)
    D1 = np.asarray(intr1["D"], dtype=np.float64).reshape(-1, 1)

    image_size = intr0.get("image_size", None)
    if not image_size or len(image_size) != 2:
        raise RuntimeError(f"image_size missing in {cam0_intr_path}")
    w, h = int(image_size[0]), int(image_size[1])

    R = np.asarray(ext["T_cj_ci"]["R"], dtype=np.float64)
    T = np.asarray(ext["T_cj_ci"]["t"], dtype=np.float64).reshape(3, 1)

    # Stereo rectification (cam0 → cam1)
    R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(
        K0, D0, K1, D1, (w, h), R, T, flags=0, alpha=0
    )

    map0x, map0y = cv2.initUndistortRectifyMap(
        K0, D0, R0, P0, (w, h), cv2.CV_32FC1
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, (w, h), cv2.CV_32FC1
    )

    # A reasonable default SGBM configuration; blockSize is set a bit larger
    # to obtain smoother disparity at the cost of some edge/detail sharpness.
    min_disp = 0
    num_disp = 128  # must be divisible by 16
    block_size = 9
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=4,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
    )

    return (map0x, map0y, map1x, map1y, Q, sgbm, min_disp, num_disp)


def list_pair_images(cam0_dir, cam1_dir, max_pairs=None):
    files0 = sorted(
        f for f in os.listdir(cam0_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    )
    files1 = sorted(
        f for f in os.listdir(cam1_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    )
    # Use intersection of file names to ensure correct pairing
    common = sorted(set(files0) & set(files1))
    if max_pairs is not None:
        common = common[:max_pairs]
    return [(os.path.join(cam0_dir, f), os.path.join(cam1_dir, f)) for f in common]


def run_visualization(
    config_path: str,
    max_pairs: int = 5,
    step: int = 1,
    show_window: bool = False,
):
    config = load_yaml(config_path)
    result_dir = config.get("result_path", None)
    if result_dir is None:
        result_dir = os.path.join(os.path.dirname(config_path), "results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    # assume camera_ids 0 and 1 exist
    cam_cfgs = config.get("camera_info", {}).get("cameras", [])
    cam0_cfg = next((c for c in cam_cfgs if c.get("camera_id") == 0), None)
    cam1_cfg = next((c for c in cam_cfgs if c.get("camera_id") == 1), None)
    if cam0_cfg is None or cam1_cfg is None:
        raise RuntimeError("camera_id 0 and 1 must be configured for stereo visualization.")

    cam0_dir = cam0_cfg["image_path"]
    cam1_dir = cam1_cfg["image_path"]
    if not os.path.isdir(cam0_dir) or not os.path.isdir(cam1_dir):
        raise RuntimeError(f"Invalid image_path for cam0 or cam1: {cam0_dir}, {cam1_dir}")

    # paths to intrinsic and extrinsic results
    cam0_intr_path = os.path.join(result_dir, "camera_0_intrinsic.yaml")
    cam1_intr_path = os.path.join(result_dir, "camera_1_intrinsic.yaml")
    stereo_extr_path = os.path.join(result_dir, "stereo_0_1_extrinsic.yaml")
    if not (os.path.isfile(cam0_intr_path) and os.path.isfile(cam1_intr_path) and os.path.isfile(stereo_extr_path)):
        raise RuntimeError(
            f"Required calibration files not found in {result_dir}: "
            "camera_0_intrinsic.yaml, camera_1_intrinsic.yaml, stereo_0_1_extrinsic.yaml"
        )

    maps_and_sgbm = build_rectification_and_sgbm(
        cam0_intr_path, cam1_intr_path, stereo_extr_path
    )
    map0x, map0y, map1x, map1y, Q, sgbm, min_disp, num_disp = maps_and_sgbm

    vis_dir = os.path.join(result_dir, "sgbm_vis")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir, exist_ok=True)

    pairs = list_pair_images(cam0_dir, cam1_dir)
    if not pairs:
        raise RuntimeError("No common image pairs found for cam0 and cam1.")

    if max_pairs is not None:
        pairs = pairs[:: max(1, step)][:max_pairs]

    print(f"[SGBM] Using {len(pairs)} stereo pairs for visualization.")

    for idx, (left_path, right_path) in enumerate(pairs):
        left = cv2.imread(left_path)
        right = cv2.imread(right_path)
        if left is None or right is None:
            continue

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Rectify
        left_rect = cv2.remap(left_gray, map0x, map0y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_gray, map1x, map1y, cv2.INTER_LINEAR)

        # Compute disparity
        disp = sgbm.compute(left_rect, right_rect).astype(np.float32) / 16.0

        # Simple disparity visualization
        disp_vis = np.zeros_like(disp, dtype=np.uint8)
        valid = disp > min_disp
        if np.any(valid):
            disp_valid = disp[valid]
            # NumPy 2.0 removed ndarray.ptp; use np.ptp instead.
            disp_range = float(np.ptp(disp_valid)) if disp_valid.size > 0 else 0.0
            if disp_range < 1e-6:
                disp_range = 1e-6
            disp_norm = np.clip(
                (disp_valid - float(disp_valid.min())) / disp_range * 255.0,
                0.0,
                255.0,
            )
            disp_vis[valid] = disp_norm.astype(np.uint8)

        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disp, Q)

        # Depth-based masking: remove negative depth and too-far points
        z = points_3d[:, :, 2]
        max_depth = 128.0  # meters (or the same unit as your calibration)

        # Build a sparse point cloud for saving
        mask = (
            valid
            & np.isfinite(z)
            & (z > 0.0)
            & (z < max_depth)
        )
        step_pc = 4
        mask[::step_pc, ::step_pc] &= True

        pts = points_3d[mask]
        cols = left[mask]

        # Build a depth image (Z) visualization, using the same depth range
        depth_img = np.zeros_like(z, dtype=np.float32)
        depth_valid = valid & np.isfinite(z) & (z > 0.0) & (z < max_depth)
        if np.any(depth_valid):
            z_valid = z[depth_valid]
            z_clipped = np.clip(z_valid, 0.0, max_depth)
            depth_norm = (z_clipped / max_depth) * 255.0
            depth_img[depth_valid] = depth_norm.astype(np.float32)
        depth_u8 = depth_img.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

        ply_path = os.path.join(vis_dir, f"cloud_{idx:03d}.ply")
        disp_path = os.path.join(vis_dir, f"disp_{idx:03d}.png")
        depth_path = os.path.join(vis_dir, f"depth_{idx:03d}.png")
        left_rect_path = os.path.join(vis_dir, f"left_rect_{idx:03d}.png")
        right_rect_path = os.path.join(vis_dir, f"right_rect_{idx:03d}.png")

        save_ply(ply_path, pts, cols)
        cv2.imwrite(disp_path, disp_color)
        cv2.imwrite(depth_path, depth_color)
        cv2.imwrite(left_rect_path, left_rect)
        cv2.imwrite(right_rect_path, right_rect)

        print(
            f"[SGBM] Pair {idx}: saved disparity to {disp_path}, depth to {depth_path} "
            f"and point cloud to {ply_path}"
        )

        # Optional interactive visualization for the first pair (or every pair if desired)
        if show_window:
            if o3d is None:
                print(
                    "[SGBM] open3d is not installed. Install it with "
                    "`pip install open3d` to enable interactive point cloud visualization."
                )
                show_window = False  # avoid repeated messages
            else:
                # Create and show an Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                # Convert BGR (OpenCV) to RGB and normalize to [0,1]
                colors_rgb = cols[:, ::-1] / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64))
                o3d.visualization.draw_geometries(
                    [pcd],
                    window_name=f"SGBM Cloud Pair {idx}",
                )
                # Only show the first by default; comment out next line if you want all
                show_window = False


def main():
    parser = argparse.ArgumentParser(
        description="Visualize stereo calibration using SGBM and 3D point cloud."
    )
    default_cfg = os.path.join(os.path.dirname(__file__), "calibration_task.yaml")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=default_cfg,
        help=f"Path to calibration_task.yaml (default: {default_cfg})",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=5,
        help="Maximum number of stereo pairs to visualize.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Stride when sampling pairs from the sequence.",
    )
    parser.add_argument(
        "--show_window",
        action="store_true",
        help="Show an interactive 3D point cloud window using Open3D (if available).",
    )
    args = parser.parse_args()

    run_visualization(
        args.config,
        max_pairs=args.max_pairs,
        step=args.step,
        show_window=args.show_window,
    )


if __name__ == "__main__":
    main()


