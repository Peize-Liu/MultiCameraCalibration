#!/usr/bin/env python3
import os
import argparse
import yaml
import cv2
import numpy as np
from scipy.optimize import least_squares

from AprilDetection.detection import Detector
from Calibrator.calibrator import IntrinsicCalibrator
from ros2_bag_utils import iter_ros2_bag_images, iter_ros2_bag_imu, ensure_dir


def load_task_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def _so3_log(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)


def _so3_exp(w: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(w.reshape(3, 1))
    return R


def _build_common_frame_points_from_two_detections(
    det_i: Detector,
    det_j: Detector,
    frame_idx_i: int,
    frame_idx_j: int,
):
    """
    Build per-frame 3D/2D correspondences for stereoCalibrate by intersecting
    tag IDs detected in both cameras.

    Returns:
        obj: (N,3) float32 in board frame
        img_i: (N,2) float32 image points in camera i
        img_j: (N,2) float32 image points in camera j
        or (None,None,None) if not enough common points.
    """
    if frame_idx_i not in det_i.results or frame_idx_j not in det_j.results:
        return None, None, None
    _, corners_i, ids_i = det_i.results[frame_idx_i]
    _, corners_j, ids_j = det_j.results[frame_idx_j]
    if corners_i is None or corners_j is None or ids_i is None or ids_j is None:
        return None, None, None
    if len(corners_i) < det_i.minimum_tag_num or len(corners_j) < det_j.minimum_tag_num:
        return None, None, None

    map_i = {int(tid[0]): c.reshape(-1, 2).astype(np.float32) for c, tid in zip(corners_i, ids_i)}
    map_j = {int(tid[0]): c.reshape(-1, 2).astype(np.float32) for c, tid in zip(corners_j, ids_j)}
    common_ids = sorted(set(map_i.keys()) & set(map_j.keys()))
    if not common_ids:
        return None, None, None

    obj = []
    img_i = []
    img_j = []
    for tid in common_ids:
        obj_tag = det_i.aprilgrid_3d_points[int(tid)].reshape(-1, 3).astype(np.float32)
        ci = map_i[tid]
        cj = map_j[tid]
        if obj_tag.shape[0] != 4 or ci.shape[0] != 4 or cj.shape[0] != 4:
            continue
        obj.extend(obj_tag)
        img_i.extend(ci)
        img_j.extend(cj)

    if not obj:
        return None, None, None
    obj = np.asarray(obj, dtype=np.float32)
    img_i = np.asarray(img_i, dtype=np.float32)
    img_j = np.asarray(img_j, dtype=np.float32)
    if obj.shape[0] < 4:
        return None, None, None
    return obj, img_i, img_j


def _pair_frames_by_time(
    det_i: Detector,
    det_j: Detector,
    tolerance_sec: float,
    max_pairs: int | None = None,
):
    """
    Pair detection frames by timestamp using a two-pointer approach.
    Returns list of (frame_idx_i, frame_idx_j, t_i, t_j).
    """
    tol = float(tolerance_sec)
    if tol < 0:
        tol = 0.0

    items_i = []
    for idx in sorted(det_i.results.keys()):
        t, corners, _ids = det_i.results[idx]
        if t is None or corners is None or len(corners) < det_i.minimum_tag_num:
            continue
        items_i.append((float(t), int(idx)))

    items_j = []
    for idx in sorted(det_j.results.keys()):
        t, corners, _ids = det_j.results[idx]
        if t is None or corners is None or len(corners) < det_j.minimum_tag_num:
            continue
        items_j.append((float(t), int(idx)))

    if not items_i or not items_j:
        return []

    pairs = []
    i = 0
    j = 0
    while i < len(items_i) and j < len(items_j):
        ti, idx_i = items_i[i]
        tj, idx_j = items_j[j]
        dt = ti - tj
        if abs(dt) <= tol:
            pairs.append((idx_i, idx_j, ti, tj))
            i += 1
            j += 1
            if max_pairs is not None and len(pairs) >= int(max_pairs):
                break
        elif dt < -tol:
            i += 1
        else:
            j += 1
    return pairs


def _draw_detected_vs_projected(img_bgr: np.ndarray, detected_xy: np.ndarray, projected_xy: np.ndarray, title: str):
    vis = img_bgr.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    det = detected_xy.reshape(-1, 2)
    proj = projected_xy.reshape(-1, 2)
    for (x, y), (u, v) in zip(det, proj):
        cv2.circle(vis, (int(round(float(x))), int(round(float(y)))), 3, (0, 255, 0), -1)
        cv2.circle(vis, (int(round(float(u))), int(round(float(v)))), 3, (0, 0, 255), -1)
        cv2.line(
            vis,
            (int(round(float(x))), int(round(float(y)))),
            (int(round(float(u))), int(round(float(v)))),
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _save_worst_frames_stereo_debug_opencv_bag(
    out_dir: str,
    frames_i: list,
    frames_j: list,
    pairs: list[tuple[int, int, float, float]],
    frames_data: list[dict],
    K_i: np.ndarray,
    D_i: np.ndarray,
    K_j: np.ndarray,
    D_j: np.ndarray,
    R_ji: np.ndarray,
    t_ji: np.ndarray,
    worst_k: int = 8,
):
    """
    Save side-by-side overlays for the worst stereo pairs (by mean reprojection error).
    Uses bag frames (in-memory images) instead of file paths.
    """
    per_frame = []
    t_ji = t_ji.reshape(3, 1)

    for f in frames_data:
        idx_i = int(f["frame_idx_i"])
        idx_j = int(f["frame_idx_j"])
        obj = f["obj"].reshape(-1, 1, 3)
        img_i = f["img_i"].reshape(-1, 1, 2)
        img_j = f["img_j"].reshape(-1, 1, 2)

        ok, rvec_i, tvec_i = cv2.solvePnP(obj, img_i, K_i, D_i, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj_i, _ = cv2.projectPoints(obj, rvec_i, tvec_i, K_i, D_i)
        err_i = np.linalg.norm(proj_i.reshape(-1, 2) - img_i.reshape(-1, 2), axis=1)

        R_i, _ = cv2.Rodrigues(rvec_i)
        t_i = tvec_i.reshape(3, 1)
        R_j = R_ji @ R_i
        t_j = R_ji @ t_i + t_ji
        rvec_j, _ = cv2.Rodrigues(R_j)
        proj_j, _ = cv2.projectPoints(obj, rvec_j, t_j, K_j, D_j)
        err_j = np.linalg.norm(proj_j.reshape(-1, 2) - img_j.reshape(-1, 2), axis=1)

        e = np.concatenate([err_i, err_j]).astype(np.float64)
        per_frame.append(
            {
                "idx_i": idx_i,
                "idx_j": idx_j,
                "mean": float(e.mean()) if e.size else 0.0,
                "max": float(e.max()) if e.size else 0.0,
                "num_points": int(e.size),
                "_proj_i": proj_i.reshape(-1, 2),
                "_proj_j": proj_j.reshape(-1, 2),
            }
        )

    per_frame.sort(key=lambda x: x["mean"], reverse=True)
    keep = per_frame[: max(1, int(worst_k))]
    ensure_dir(out_dir)
    meta = []

    # Map from idx->image
    img_map_i = {k: frames_i[k][1] for k in range(len(frames_i))}
    img_map_j = {k: frames_j[k][1] for k in range(len(frames_j))}

    for rank, item in enumerate(keep):
        idx_i = int(item["idx_i"])
        idx_j = int(item["idx_j"])
        if idx_i not in img_map_i or idx_j not in img_map_j:
            continue
        im0 = img_map_i[idx_i].copy()
        im1 = img_map_j[idx_j].copy()

        f = next((x for x in frames_data if int(x["frame_idx_i"]) == idx_i and int(x["frame_idx_j"]) == idx_j), None)
        if f is None:
            continue
        det0 = f["img_i"].reshape(-1, 2)
        det1 = f["img_j"].reshape(-1, 2)

        vis0 = _draw_detected_vs_projected(
            im0,
            det0,
            item["_proj_i"],
            f"cam{int(0)} idx {idx_i} mean={item['mean']:.2f}px max={item['max']:.2f}px",
        )
        vis1 = _draw_detected_vs_projected(
            im1,
            det1,
            item["_proj_j"],
            f"cam{int(1)} idx {idx_j} mean={item['mean']:.2f}px max={item['max']:.2f}px",
        )

        # pad to same height
        h0, _w0 = vis0.shape[:2]
        h1, _w1 = vis1.shape[:2]
        H = max(h0, h1)
        if h0 != H:
            vis0 = cv2.copyMakeBorder(vis0, 0, H - h0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if h1 != H:
            vis1 = cv2.copyMakeBorder(vis1, 0, H - h1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        combo = np.concatenate([vis0, vis1], axis=1)
        out_path = os.path.join(out_dir, f"{rank:03d}_i{idx_i:06d}_j{idx_j:06d}.png")
        cv2.imwrite(out_path, combo)
        meta.append(
            {
                "rank": int(rank),
                "idx_i": idx_i,
                "idx_j": idx_j,
                "mean": float(item["mean"]),
                "max": float(item["max"]),
                "num_points": int(item["num_points"]),
            }
        )
    return meta


def _integrate_gyro_delta_R(imu_ts: np.ndarray, imu_gyro: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """
    Integrate gyro between [t1, t2] (seconds) into a delta rotation matrix.
    Simple piecewise-constant / trapezoidal integration on SO(3) via small-angle exp.
    """
    if t2 <= t1:
        return np.eye(3, dtype=np.float64)

    # find samples in window (with 1-sample padding)
    idx = np.where((imu_ts >= t1) & (imu_ts <= t2))[0]
    if idx.size < 2:
        # not enough IMU samples; return identity
        return np.eye(3, dtype=np.float64)

    R = np.eye(3, dtype=np.float64)
    for k in range(int(idx[0]), int(idx[-1])):
        ta = float(imu_ts[k])
        tb = float(imu_ts[k + 1])
        if tb <= t1:
            continue
        if ta >= t2:
            break
        a = max(ta, t1)
        b = min(tb, t2)
        dt = b - a
        if dt <= 0:
            continue
        wa = imu_gyro[k]
        wb = imu_gyro[k + 1]
        w = 0.5 * (wa + wb)
        R = _so3_exp(w * dt) @ R
    return R


def _estimate_imu_cam_rotation_and_dt(
    cam_times: np.ndarray,
    cam_R_delta: list[np.ndarray],
    imu_ts: np.ndarray,
    imu_gyro: np.ndarray,
    dt_init: float,
    dt_bounds: tuple[float, float],
    min_pairs: int = 30,
):
    """
    Estimate R_cam_imu and dt (t_cam ≈ t_imu + dt) using rotation-only alignment:
      R_cam_delta(k) ≈ R_cam_imu * R_imu_delta(k, dt) * R_cam_imu^T
    """
    if len(cam_R_delta) < min_pairs:
        raise RuntimeError(f"Not enough camera rotation pairs for IMU calibration: {len(cam_R_delta)} < {min_pairs}")

    dt_lo, dt_hi = float(dt_bounds[0]), float(dt_bounds[1])

    def residual(x):
        w = x[:3]
        dt = float(x[3])
        R_ci = _so3_exp(w)
        r = []
        for k, R_cam in enumerate(cam_R_delta):
            t1 = float(cam_times[k] - dt)
            t2 = float(cam_times[k + 1] - dt)
            R_imu = _integrate_gyro_delta_R(imu_ts, imu_gyro, t1, t2)
            R_pred = R_ci @ R_imu @ R_ci.T
            R_err = R_cam @ R_pred.T
            r.append(_so3_log(R_err))
        return np.concatenate(r).astype(np.float64)

    x0 = np.array([0.0, 0.0, 0.0, float(dt_init)], dtype=np.float64)
    bounds = (
        np.array([-np.inf, -np.inf, -np.inf, dt_lo], dtype=np.float64),
        np.array([+np.inf, +np.inf, +np.inf, dt_hi], dtype=np.float64),
    )
    res = least_squares(residual, x0, method="trf", bounds=bounds, verbose=1)
    w_opt = res.x[:3]
    dt_opt = float(res.x[3])
    R_cam_imu = _so3_exp(w_opt)
    return R_cam_imu, dt_opt, res


def calibrate_camera_from_bag(
    cam_cfg: dict,
    tag_yaml: str,
    result_dir: str,
    bag_path: str,
):
    cam_id = int(cam_cfg["camera_id"])
    cam_model = str(cam_cfg.get("camera_model", "pinhole"))
    image_topic = str(cam_cfg["image_topic"])
    image_type = str(cam_cfg.get("image_type", "raw"))
    max_frames = cam_cfg.get("max_frames", None)
    stride = int(cam_cfg.get("stride", 1))

    detector = Detector(
        camera_id=cam_id,
        tag_config="tag36h11",
        minimum_tag_num=4,
        yaml_file=tag_yaml,
    )

    frames = []
    shape = None
    img_idx = 0
    for frame in iter_ros2_bag_images(bag_path, image_topic, image_type=image_type, max_frames=max_frames, stride=stride):
        img = frame.img_bgr
        if img is None:
            continue
        if shape is None:
            shape = img.shape[:2]
        # store timestamp in results for downstream IMU alignment
        detector.detect(img, frame.t_sec, img_idx, show=False, enable_subpix=True)
        frames.append((frame.t_sec, img))
        img_idx += 1

    if shape is None:
        raise RuntimeError(f"camera_id={cam_id}: failed to read images from bag topic {image_topic}")

    h, w = shape
    result = {"camera_id": cam_id, "camera_model": cam_model, "image_size": [w, h]}

    used_indices = sorted(detector.results.keys())
    if cam_model == "omni":
        calibrator = IntrinsicCalibrator()
        retval, K, xi, D, rvecs, tvecs, idx = calibrator.calibrate_mono(
            detector, (h, w), None, None, selected_indices=used_indices
        )
        if retval is None:
            raise RuntimeError(f"camera_id={cam_id}: intrinsic calibration failed.")
        result.update({"rmse": float(retval), "K": K.tolist(), "xi": float(xi[0]), "D": D.reshape(-1).tolist()})
        return detector, frames, result

    if cam_model == "pinhole":
        obj_pts, img_pts = detector.gather_information(selected_indices=used_indices)
        retval, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)
        result.update({"rmse": float(retval), "K": K.tolist(), "D": D.reshape(-1).tolist()})
        return detector, frames, result

    raise ValueError(f"Unsupported camera_model: {cam_model}")


def run_stereo_calibration_from_bag(cfg: dict, result_dir: str, detectors: dict, frames_map: dict, intrinsics: dict):
    extr = cfg.get("extrinsic_info", {})
    if not extr or not bool(extr.get("enable", False)):
        print("[Stereo][Bag] extrinsic_info.enable is False, skip stereo calibration.")
        return

    method = str(extr.get("method", "opencv")).lower()
    if method != "opencv":
        print("[Stereo][Bag] Only method=opencv is implemented in bag pipeline for now.")
        return

    pairs = extr.get("camera_pairs", [])
    if not pairs:
        print("[Stereo][Bag] No camera_pairs configured, skip stereo calibration.")
        return

    tol = float(extr.get("time_sync_tolerance_sec", 0.01))
    max_pairs = extr.get("max_pairs", None)
    save_worst = bool(extr.get("save_worst_frames", False))
    worst_k = int(extr.get("worst_k", 8))

    # OpenCV termination criteria
    crit = None
    oc = cfg.get("opencv_criteria")
    if isinstance(oc, dict):
        max_iter = int(oc.get("max_iter", 0))
        eps = float(oc.get("eps", 0.0))
        if max_iter > 0 and eps > 0:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

    for ci, cj in pairs:
        ci = int(ci)
        cj = int(cj)
        if ci not in detectors or cj not in detectors:
            print(f"[Stereo][Bag] Missing detectors for pair ({ci},{cj}), skip.")
            continue
        if ci not in intrinsics or cj not in intrinsics:
            print(f"[Stereo][Bag] Missing intrinsics for pair ({ci},{cj}), skip.")
            continue

        intr_i = intrinsics[ci]
        intr_j = intrinsics[cj]
        if intr_i.get("camera_model") != "pinhole" or intr_j.get("camera_model") != "pinhole":
            print(f"[Stereo][Bag] Only pinhole-pinhole pairs are supported. Skip ({ci},{cj}).")
            continue

        K_i = np.asarray(intr_i["K"], dtype=np.float64)
        D_i = np.asarray(intr_i["D"], dtype=np.float64).reshape(-1, 1)
        K_j = np.asarray(intr_j["K"], dtype=np.float64)
        D_j = np.asarray(intr_j["D"], dtype=np.float64).reshape(-1, 1)
        img_w, img_h = int(intr_i["image_size"][0]), int(intr_i["image_size"][1])

        det_i = detectors[ci]
        det_j = detectors[cj]

        matched = _pair_frames_by_time(det_i, det_j, tolerance_sec=tol, max_pairs=max_pairs)
        if not matched:
            print(f"[Stereo][Bag] No matched frames within tol={tol}s for pair ({ci},{cj}), skip.")
            continue

        frames_data = []
        for idx_i, idx_j, ti, tj in matched:
            obj, img_i, img_j = _build_common_frame_points_from_two_detections(det_i, det_j, idx_i, idx_j)
            if obj is None:
                continue
            frames_data.append(
                {
                    "frame_idx_i": int(idx_i),
                    "frame_idx_j": int(idx_j),
                    "t_i": float(ti),
                    "t_j": float(tj),
                    "obj": obj,
                    "img_i": img_i,
                    "img_j": img_j,
                    "score": int(obj.shape[0]),
                }
            )

        if not frames_data:
            print(f"[Stereo][Bag] No usable matched frames for pair ({ci},{cj}), skip.")
            continue

        # Use the best frames (by common points) if too many
        max_frames = int(extr.get("max_frames", 200))
        if len(frames_data) > max_frames:
            frames_data.sort(key=lambda f: f["score"], reverse=True)
            frames_data = frames_data[:max_frames]
            frames_data.sort(key=lambda f: (f["frame_idx_i"], f["frame_idx_j"]))
            print(f"[Stereo][Bag] Pair ({ci},{cj}): using top {max_frames} frames by common tag points.")

        obj_pts = [f["obj"].reshape(-1, 1, 3) for f in frames_data]
        img_pts_i = [f["img_i"].reshape(-1, 1, 2) for f in frames_data]
        img_pts_j = [f["img_j"].reshape(-1, 1, 2) for f in frames_data]

        flags = cv2.CALIB_FIX_INTRINSIC
        print(f"[Stereo][Bag][OpenCV] stereoCalibrate pair ({ci},{cj}) with {len(frames_data)} frames.")
        rms, _Ki, _Di, _Kj, _Dj, R, T, E, F = cv2.stereoCalibrate(
            obj_pts,
            img_pts_i,
            img_pts_j,
            K_i,
            D_i,
            K_j,
            D_j,
            (img_w, img_h),
            criteria=crit,
            flags=flags,
        )

        # Evaluate reprojection error per point (solvePnP on cam i for each view)
        per_point_err = []
        for f in frames_data:
            obj = f["obj"].reshape(-1, 1, 3)
            img_i_v = f["img_i"].reshape(-1, 1, 2)
            ok, rvec_i, tvec_i = cv2.solvePnP(obj, img_i_v, K_i, D_i, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            proj_i, _ = cv2.projectPoints(obj, rvec_i, tvec_i, K_i, D_i)
            per_point_err.append(np.linalg.norm(proj_i.reshape(-1, 2) - f["img_i"], axis=1))

            R_i, _ = cv2.Rodrigues(rvec_i)
            t_i = tvec_i.reshape(3, 1)
            R_j = np.asarray(R, dtype=np.float64) @ R_i
            t_j = np.asarray(R, dtype=np.float64) @ t_i + np.asarray(T, dtype=np.float64).reshape(3, 1)
            rvec_j, _ = cv2.Rodrigues(R_j)
            proj_j, _ = cv2.projectPoints(obj, rvec_j, t_j, K_j, D_j)
            per_point_err.append(np.linalg.norm(proj_j.reshape(-1, 2) - f["img_j"], axis=1))

        if per_point_err:
            e = np.concatenate(per_point_err).astype(np.float64)
            err_mean = float(e.mean())
            err_min = float(e.min())
            err_max = float(e.max())
            num_points = int(e.size)
        else:
            err_mean = err_min = err_max = 0.0
            num_points = 0

        out_ex_path = os.path.join(result_dir, f"stereo_{ci}_{cj}_extrinsic_opencv.yaml")
        out_data = {
            "pair": [int(ci), int(cj)],
            "method": "opencv_stereoCalibrate",
            "time_sync_tolerance_sec": float(tol),
            "num_frames": int(len(frames_data)),
            "T_cj_ci": {"R": np.asarray(R, dtype=np.float64).tolist(), "t": np.asarray(T, dtype=np.float64).reshape(3).tolist()},
            "opencv": {"rms": float(rms), "flags": int(flags)},
            "reprojection_error": {"mean": err_mean, "min": err_min, "max": err_max, "num_points": num_points},
        }

        if save_worst:
            out_dir = os.path.join(result_dir, f"stereo_{ci}_{cj}_worst_frames")
            worst = _save_worst_frames_stereo_debug_opencv_bag(
                out_dir,
                frames_map[ci],
                frames_map[cj],
                matched,
                frames_data,
                K_i,
                D_i,
                K_j,
                D_j,
                np.asarray(R, dtype=np.float64),
                np.asarray(T, dtype=np.float64),
                worst_k=worst_k,
            )
            if worst:
                out_data["worst_frames_dir"] = os.path.basename(out_dir)
                out_data["worst_frames"] = worst

        with open(out_ex_path, "w") as f:
            yaml.safe_dump(out_data, f)
        print(
            f"[Stereo][Bag][OpenCV] Pair ({ci},{cj}) saved: {out_ex_path} "
            f"(rms={float(rms):.4f}, mean={err_mean:.4f}px, max={err_max:.4f}px)"
        )


def main():
    default_cfg = os.path.join(os.path.dirname(__file__), "calibration_bag_task.yaml")
    parser = argparse.ArgumentParser(description="Multi-camera calibration from ROS2 bag")
    parser.add_argument("--config", "-c", type=str, default=default_cfg)
    args = parser.parse_args()

    cfg = load_task_config(args.config)
    result_dir = cfg.get("result_path") or os.path.join(os.path.dirname(__file__), "bag_results")
    ensure_dir(result_dir)

    tag_info = cfg.get("tag_info", {})
    if isinstance(tag_info, dict) and tag_info.get("target_type", "aprilgrid") == "aprilgrid":
        tag_yaml = os.path.join(result_dir, "aprilgrid_from_task.yaml")
        with open(tag_yaml, "w") as f:
            yaml.safe_dump(tag_info, f)
    else:
        tag_yaml = os.path.join(os.path.dirname(__file__), "april_6x6.yaml")

    bag_info = cfg.get("bag_info", {})
    bag_path = str(bag_info.get("bag_path", ""))
    if not bag_path:
        raise RuntimeError("bag_info.bag_path is required")

    cam_cfgs = cfg.get("camera_info", {}).get("cameras", [])
    if not cam_cfgs:
        raise RuntimeError("camera_info.cameras is empty")

    detectors = {}
    frames_map = {}
    intrinsics = {}

    # 1) Mono calibration for each camera
    for cam_cfg in cam_cfgs:
        det, frames, intr = calibrate_camera_from_bag(cam_cfg, tag_yaml, result_dir, bag_path)
        cid = int(cam_cfg["camera_id"])
        detectors[cid] = det
        frames_map[cid] = frames
        intrinsics[cid] = intr
        out_path = os.path.join(result_dir, f"camera_{cid}_intrinsic.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump(intr, f)
        print(f"[Intrinsic][Bag] camera_id={cid} saved: {out_path}")

    # 2) Stereo extrinsic from bag (OpenCV stereoCalibrate)
    run_stereo_calibration_from_bag(cfg, result_dir, detectors, frames_map, intrinsics)

    # 3) IMU extrinsic (rotation) + dt with base_camera_id
    imu_cfg = cfg.get("imu_info", {})
    if imu_cfg and bool(imu_cfg.get("enable", False)):
        base_id = int(imu_cfg.get("base_camera_id", cfg.get("extrinsic_info", {}).get("base_camera_id", 0)))
        imu_topic = str(imu_cfg.get("imu_topic", ""))
        if not imu_topic:
            raise RuntimeError("imu_info.imu_topic is required when imu_info.enable is true")
        if base_id not in detectors or base_id not in intrinsics:
            raise RuntimeError(f"base_camera_id={base_id} missing from calibrated cameras")
        if intrinsics[base_id].get("camera_model") != "pinhole":
            raise RuntimeError("IMU alignment currently implemented for pinhole base camera only")

        # Build camera rotation deltas from board PnP poses
        det = detectors[base_id]
        K = np.asarray(intrinsics[base_id]["K"], dtype=np.float64)
        D = np.asarray(intrinsics[base_id]["D"], dtype=np.float64).reshape(-1, 1)

        # Use all frames that have detections
        idxs = sorted(det.results.keys())
        cam_times = []
        cam_R_cb = []
        for fid in idxs:
            t_sec, corners_list, ids = det.results[fid]
            if t_sec is None:
                continue
            # build points for this frame
            obj = []
            img = []
            for corners, tag_id in zip(corners_list, ids):
                obj.extend(det.aprilgrid_3d_points[int(tag_id[0])].reshape(-1, 3))
                img.extend(corners.reshape(-1, 2))
            obj = np.asarray(obj, dtype=np.float32).reshape(-1, 1, 3)
            img = np.asarray(img, dtype=np.float32).reshape(-1, 1, 2)
            if obj.shape[0] < 4:
                continue
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            T = _rt_to_T(rvec, tvec)
            cam_times.append(float(t_sec))
            cam_R_cb.append(T[:3, :3])

        cam_times = np.asarray(cam_times, dtype=np.float64)
        if cam_times.size < 3:
            raise RuntimeError("Not enough camera poses to align with IMU")

        cam_R_delta = []
        for k in range(cam_times.size - 1):
            R1 = cam_R_cb[k]
            R2 = cam_R_cb[k + 1]
            cam_R_delta.append(R2 @ R1.T)

        # Read IMU
        imu_samples = list(iter_ros2_bag_imu(bag_path, imu_topic))
        imu_ts = np.asarray([s.t_sec for s in imu_samples], dtype=np.float64)
        imu_gyro = np.asarray([s.gyro_rad_s for s in imu_samples], dtype=np.float64)

        dt_init = float(imu_cfg.get("dt_init", 0.0))
        dt_bounds = tuple(imu_cfg.get("dt_bounds", [-0.2, 0.2]))
        min_pairs = int(imu_cfg.get("min_pairs", 30))

        R_cam_imu, dt_opt, opt = _estimate_imu_cam_rotation_and_dt(
            cam_times=cam_times,
            cam_R_delta=cam_R_delta,
            imu_ts=imu_ts,
            imu_gyro=imu_gyro,
            dt_init=dt_init,
            dt_bounds=dt_bounds,
            min_pairs=min_pairs,
        )

        out = {
            "base_camera_id": int(base_id),
            "imu_topic": imu_topic,
            "t_cam_minus_t_imu": float(dt_opt),
            "R_cam_imu": R_cam_imu.tolist(),
            "t_cam_imu": [0.0, 0.0, 0.0],  # translation is not estimated in this rotation-only method
            "optimization": {
                "cost": float(opt.cost),
                "num_iterations": int(opt.nfev),
            },
            "note": "Rotation+dt only. Translation is not estimated from raw IMU without a full VIO/position constraint.",
        }
        out_path = os.path.join(result_dir, "imu_extrinsic.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump(out, f)
        print(f"[IMU][Bag] Saved: {out_path}")


if __name__ == "__main__":
    main()


