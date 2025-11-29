#!/usr/bin/env python3
import os
import argparse
import yaml
import cv2
import numpy as np

from AprilDetection.detection import Detector
from Calibrator.calibrator import IntrinsicCalibrator

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x


def load_task_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class CoverageTracker:
    def __init__(self, image_shape, grid_downsample: int = 1):
        h, w = image_shape
        self.down = max(1, int(grid_downsample))
        self.h = h // self.down
        self.w = w // self.down
        self.cover_mask = np.zeros((self.h, self.w), dtype=bool)

    def _rasterize_polygon(self, poly_pts: np.ndarray) -> np.ndarray:
        """
        将像素坐标多边形栅格化到下采样网格上，返回 bool mask。
        """
        if poly_pts is None or len(poly_pts) < 3:
            return np.zeros_like(self.cover_mask, dtype=bool)
        pts_ds = (poly_pts / float(self.down)).astype(np.int32)
        mask = np.zeros_like(self.cover_mask, dtype=np.uint8)
        cv2.fillPoly(mask, [pts_ds], 1)
        return mask.astype(bool)

    def update_and_check(self, poly_pts: np.ndarray, added_ratio_thresh: float):
        """
        更新覆盖并判断该帧是否带来足够新的覆盖区域。

        Returns:
            added_ratio: 本帧新增区域占本帧棋盘区域的比例
            global_ratio: 全局覆盖比例（相对于整幅图）
            accept: 是否接受该帧
        """
        frame_mask = self._rasterize_polygon(poly_pts)
        board_area = frame_mask.sum()
        if board_area == 0:
            return 0.0, float(self.cover_mask.mean()), False

        new_pixels = frame_mask & (~self.cover_mask)
        new_count = new_pixels.sum()
        added_ratio = float(new_count) / float(board_area)

        accept = added_ratio > added_ratio_thresh
        if accept:
            self.cover_mask |= frame_mask

        global_ratio = float(self.cover_mask.mean())
        return added_ratio, global_ratio, accept


def compute_board_polygon(marker_corners, ids):
    """
    根据一帧的检测结果估计标定板在图像中的外接区域（像素多边形）。
    简化实现：对所有 tag 角点做 convex hull。
    """
    if marker_corners is None or len(marker_corners) == 0:
        return None
    all_pts = []
    for corners in marker_corners:
        c = corners.reshape(-1, 2)
        all_pts.append(c)
    if not all_pts:
        return None
    pts = np.vstack(all_pts).astype(np.float32)
    hull = cv2.convexHull(pts)
    return hull.reshape(-1, 2)


def calibrate_single_camera(cam_cfg: dict,
                            tag_yaml: str,
                            result_dir: str,
                            verbose: bool = False,
                            coverage_cfg: dict | None = None):
    cam_id = cam_cfg["camera_id"]
    cam_model = cam_cfg.get("camera_model", "omni")
    img_dir = cam_cfg.get("image_path")
    sub_pix_predict = cam_cfg.get("sub_pix_predict", False)

    if img_dir is None:
        raise ValueError(f"camera_id={cam_id} 的 image_path 未配置")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"camera_id={cam_id} 的图像目录不存在: {img_dir}")

    # AprilTag 检测器
    detector = Detector(
        camera_id=cam_id,
        tag_config="tag36h11",
        minimum_tag_num=4,
        yaml_file=tag_yaml,
    )

    # 读取图像并进行角点检测
    image_files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    )

    if not image_files:
        raise RuntimeError(f"camera_id={cam_id} 在目录 {img_dir} 中没有找到图像")

    images = []
    shape = None
    img_idx = 0
    for fname in tqdm(image_files, desc=f"Cam {cam_id} detect"):
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        if shape is None:
            shape = img.shape[:2]
        # 这里默认开启亚像素角点（与 TestCalib 中的一致）
        _, _ = detector.detect(img, None, img_idx, show=False, enable_subpix=True)
        images.append(img)
        img_idx += 1

    if shape is None:
        raise RuntimeError(f"camera_id={cam_id} 所有图像读取失败，无法标定")

    h, w = shape

    result = {
        "camera_id": cam_id,
        "camera_model": cam_model,
        "image_size": [w, h],
    }

    # 覆盖筛选：根据 coverage_filter 配置决定是否启用
    coverage_enable = False
    added_ratio_thresh = 0.0
    grid_downsample = 1
    if coverage_cfg is not None:
        coverage_enable = bool(coverage_cfg.get("enable", False))
        added_ratio_thresh = float(coverage_cfg.get("added_ratio_thresh", 0.0))
        grid_downsample = int(coverage_cfg.get("grid_downsample", 1))

    all_indices = sorted(detector.results.keys())
    used_indices = all_indices.copy()
    coverage_ratio = 0.0

    if coverage_enable and all_indices:
        tracker = CoverageTracker(shape, grid_downsample=grid_downsample)
        used_indices = []
        for idx in all_indices:
            _, marker_corners, ids = detector.results[idx]
            poly = compute_board_polygon(marker_corners, ids)
            added_ratio, coverage_ratio, accept = tracker.update_and_check(
                poly, added_ratio_thresh
            )
            if accept:
                used_indices.append(idx)

        if not used_indices:
            # 如果全部被筛掉，为了保证可标定，退回使用全部帧
            used_indices = all_indices

    total_frames = len(all_indices)
    used_frames = len(used_indices)

    if cam_model == "omni":
        # 使用现有的 omnidir 单目标定器得到初始结果
        calibrator = IntrinsicCalibrator()
        retval_init, K, xi, D, rvecs, tvecs, idx = calibrator.calibrate_mono(
            detector, (h, w), None, None, selected_indices=used_indices
        )
        if retval_init is None:
            raise RuntimeError(f"camera_id={cam_id} 内参标定失败（点数不足）")

        retval = float(retval_init)

        # 如果开启 sub_pix_predict，则调用 refine_calibration 做两轮预测+亚像素优化+重标定
        if sub_pix_predict:
            detection_result = detector.results
            valid_detection_result = []
            valid_detected_images = []
            used_indices_for_refine = idx[0] if hasattr(idx, "__getitem__") else idx
            for i in used_indices_for_refine:
                valid_detection_result.append(detection_result[i])
                valid_detected_images.append(images[i])

            # 进行两轮 refine_calibration + omnidir.calibrate
            for iter_idx in (1, 2):
                debug_save_dir = None
                visualize = False
                if verbose:
                    debug_base = os.path.join(os.getcwd(), "debug", f"camera_{cam_id}")
                    debug_save_dir = os.path.join(debug_base, f"iter_{iter_idx}")
                    visualize = False  # 如需弹窗查看可改为 True

                points_2d, points_3d = calibrator.refine_calibration(
                    valid_detection_result=valid_detection_result,
                    rvecs=rvecs,
                    tvecs=tvecs,
                    K=K,
                    xi=xi,
                    D=D,
                    sub_pix_window_size=5,
                    valid_detected_images=valid_detected_images,
                    aprilgrid_3d_points=detector.aprilgrid_3d_points,
                    image_shape=shape,
                    ooi_threshold=0.75,
                    debug_save_dir=debug_save_dir,
                    visualize=visualize,
                )
                flags = cv2.omnidir.CALIB_USE_GUESS
                retval_refine, K, xi, D, rvecs, tvecs, idx = cv2.omnidir.calibrate(
                    points_3d,
                    points_2d,
                    shape,
                    K,
                    xi,
                    D,
                    flags=flags,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        200,
                        1e-6,
                    ),
                )
                retval = float(retval_refine)

        result.update(
            {
                "rmse": float(retval),
                "rmse_init": float(retval_init),
                "K": K.tolist(),
                "xi": float(xi[0]),
                "D": D.reshape(-1).tolist(),
            }
        )

        result["sub_pix_predict_used"] = bool(sub_pix_predict)

    elif cam_model == "pinhole":
        # 标准针孔相机模型标定
        obj_pts, img_pts = detector.gather_information()
        if not obj_pts or not img_pts:
            raise RuntimeError(f"camera_id={cam_id} 内参标定失败（点数不足）")
        print("receive obj_pts and img_pts start to calibrate Camera Intrinsic should wait for a while")
        retval, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts, img_pts, (w, h), None, None
        )
        result.update(
            {
                "rmse": float(retval),
                "K": K.tolist(),
                "D": D.reshape(-1).tolist(),
            }
        )
        result["sub_pix_predict_used"] = False  # 当前只对 omni 做高级预测流程
    else:
        raise ValueError(f"不支持的 camera_model: {cam_model}")

    # 覆盖统计信息写入结果
    result["used_frames"] = int(used_frames)
    result["total_frames"] = int(total_frames)
    result["coverage_ratio"] = float(coverage_ratio)
    result["coverage_added_thresh"] = float(added_ratio_thresh)
    result["coverage_grid_downsample"] = int(grid_downsample)

    # 保存结果
    ensure_dir(result_dir)
    out_path = os.path.join(result_dir, f"camera_{cam_id}_intrinsic.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(result, f)
    print(f"[Intrinsic] camera_id={cam_id} 标定完成，结果已保存到: {out_path}")


def run_extrinsic_calibration(config: dict, result_dir: str):
    """
    外参标定占位函数（当前版本完全不做任何外参计算）。

    你的计划是：
      1. 先用 PnP（solvePnP / omnidir.solvePnP）从共视 AprilGrid 帧估计每个相机的初始位姿；
      2. 再构建非线性优化器，引入多相机之间的相对位姿约束和环路约束，联合优化外参。

    为了先专注验证多相机内参标定和结果输出，本函数暂时留空。
    后续实现外参时，可以在这里接入相机对、base_camera_id 和环路优化逻辑。
    """
    return


def main():
    default_cfg = os.path.join(os.path.dirname(__file__), "calibration_task.yaml")
    parser = argparse.ArgumentParser(description="Multi-camera intrinsic calibration task")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=default_cfg,
        help=f"标定任务配置文件路径（默认: {default_cfg}）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="保存 sub-pixel 优化的调试图片到 ./debug/camera_id/ 下",
    )
    args = parser.parse_args()

    config = load_task_config(args.config)

    tag_info = config.get("tag_info", {})
    # 当前实现继续使用独立的 AprilGrid 配置文件（例如 april_6x6.yaml）
    # 如果你希望完全由 tag_info 驱动，可以在 Detector 内部做进一步扩展。
    tag_yaml = os.path.join(
        os.path.dirname(__file__), "april_6x6.yaml"
    )

    result_dir = config.get("result_path")
    if result_dir is None:
        result_dir = os.path.join(os.path.dirname(__file__), "results")

    cam_cfgs = config.get("camera_info", {}).get("cameras", [])
    coverage_cfg = config.get("coverage_filter", {})
    if not cam_cfgs:
        raise RuntimeError("camera_info.cameras 列表为空，请在 calibration_task.yaml 中配置相机。")

    # 逐路相机做内参标定
    for cam_cfg in cam_cfgs:
        calibrate_single_camera(
            cam_cfg,
            tag_yaml,
            result_dir,
            verbose=args.verbose,
            coverage_cfg=coverage_cfg,
        )

    # 外参标定已按你的要求暂时完全关闭
    # run_extrinsic_calibration(config, result_dir)


if __name__ == "__main__":
    main()


