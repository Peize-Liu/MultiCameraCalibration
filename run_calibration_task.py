#!/usr/bin/env python3
import os
import argparse
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from scipy.optimize import least_squares

from AprilDetection.detection import Detector
from Calibrator.calibrator import IntrinsicCalibrator

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
        # store original image size for visualization
        self.full_h = h
        self.full_w = w
        # boolean mask on the downsampled grid indicating covered area
        self.cover_mask = np.zeros((self.h, self.w), dtype=bool)

    def _rasterize_polygon(self, poly_pts: np.ndarray) -> np.ndarray:
        """
        Rasterize a polygon given in pixel coordinates onto the downsampled grid.

        Returns:
            A bool mask on the downsampled grid indicating the polygon area.
        """
        if poly_pts is None or len(poly_pts) < 3:
            return np.zeros_like(self.cover_mask, dtype=bool)
        pts_ds = (poly_pts / float(self.down)).astype(np.int32)
        mask = np.zeros_like(self.cover_mask, dtype=np.uint8)
        cv2.fillPoly(mask, [pts_ds], 1)
        return mask.astype(bool)

    def update_and_check(self, poly_pts: np.ndarray, added_ratio_thresh: float):
        """
        Update coverage with the current frame and decide whether to accept it.

        Returns:
            added_ratio: ratio of newly covered area w.r.t. the board area in this frame
            global_ratio: global coverage ratio w.r.t. the whole image
            accept: whether this frame is accepted by the coverage filter
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

    def save_coverage_image(self, out_path: str, upscale_to_full: bool = True):
        """
        Save a visualization image of the accumulated coverage mask.

        Args:
            out_path: where to save the visualization image.
            upscale_to_full: if True, resize the downsampled mask back to full image size.
        """
        # convert bool mask to 0/255 uint8
        img = (self.cover_mask.astype(np.uint8) * 255)
        if upscale_to_full:
            img = cv2.resize(
                img,
                (self.full_w, self.full_h),
                interpolation=cv2.INTER_NEAREST,
            )
        # apply a colormap for better visualization
        color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite(out_path, color)


def compute_board_polygon(marker_corners, ids):
    """
    Estimate the outer polygon of the calibration board in the image
    from the detected tag corners of a single frame.

    Simplified implementation: compute a convex hull over all tag corners.
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


def _save_detected_corner_order_debug(
    out_path: str,
    image: np.ndarray,
    marker_corners,
    ids,
    max_tags: int = 6,
):
    """
    Save a debug visualization to verify the 2D corner ordering returned by OpenCV.

    It draws each detected tag with its 4 corners labeled 0..3.
    This is useful when RMSE is unexpectedly high and you suspect a 2D/3D corner
    order mismatch.
    """
    if image is None or marker_corners is None or ids is None:
        return False
    if len(marker_corners) == 0:
        return False

    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    n = min(int(len(marker_corners)), int(max_tags))
    for k in range(n):
        corners = marker_corners[k].reshape(-1, 2)
        tag_id = int(ids[k][0]) if hasattr(ids[k], "__len__") else int(ids[k])
        # draw polygon
        pts_i32 = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts_i32], True, (0, 255, 0), 2)
        # draw corner indices
        for i, (x, y) in enumerate(corners):
            cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"{tag_id}:{i}",
                (int(x) + 6, int(y) - 6),
                font,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, vis)
    return True


def _flatten_obj_img_points(obj_pts: np.ndarray, img_pts: np.ndarray):
    """
    Normalize object/image points shapes to:
      obj: (N, 3) float64
      img: (N, 2) float64
    Accepts common OpenCV calibration shapes like (N,1,3), (N,3), (M,4,3) etc.
    """
    obj = np.asarray(obj_pts)
    img = np.asarray(img_pts)
    obj = obj.reshape(-1, 3).astype(np.float64)
    img = img.reshape(-1, 2).astype(np.float64)
    return obj, img


def _summarize_reprojection_errors(e_all: np.ndarray) -> dict:
    """
    Summarize per-point reprojection errors (L2 in pixels).

    Returns a dict containing robust percentiles in addition to mean/min/max.
    """
    e = np.asarray(e_all, dtype=np.float64).reshape(-1)
    if e.size == 0:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "num_points": 0,
        }
    return {
        "mean": float(e.mean()),
        "min": float(e.min()),
        "max": float(e.max()),
        "p50": float(np.percentile(e, 50)),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "num_points": int(e.size),
    }


def _compute_reprojection_error_stats_pinhole(
    obj_pts_list,
    img_pts_list,
    rvecs,
    tvecs,
    K: np.ndarray,
    D: np.ndarray,
):
    """Return summary dict of per-point reprojection error (L2 in pixels)."""
    errs_chunks = []
    if obj_pts_list is None or img_pts_list is None:
        return _summarize_reprojection_errors(np.zeros(0, dtype=np.float64))
    for obj_pts, img_pts, rvec, tvec in zip(obj_pts_list, img_pts_list, rvecs, tvecs):
        obj, img = _flatten_obj_img_points(obj_pts, img_pts)
        proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)
        e = np.linalg.norm(proj - img, axis=1)
        errs_chunks.append(e)
    if not errs_chunks:
        return _summarize_reprojection_errors(np.zeros(0, dtype=np.float64))
    e_all = np.concatenate(errs_chunks)
    return _summarize_reprojection_errors(e_all)


def _compute_reprojection_error_stats_omni(
    obj_pts_list,
    img_pts_list,
    rvecs,
    tvecs,
    K: np.ndarray,
    xi: float,
    D: np.ndarray,
):
    """Return summary dict of per-point reprojection error (L2 in pixels)."""
    errs_chunks = []
    if obj_pts_list is None or img_pts_list is None:
        return _summarize_reprojection_errors(np.zeros(0, dtype=np.float64))
    xi_f = float(xi)
    for obj_pts, img_pts, rvec, tvec in zip(obj_pts_list, img_pts_list, rvecs, tvecs):
        obj, img = _flatten_obj_img_points(obj_pts, img_pts)
        proj, _ = cv2.omnidir.projectPoints(
            obj.reshape(-1, 1, 3),
            rvec,
            tvec,
            K,
            xi_f,
            D,
        )
        proj = proj.reshape(-1, 2)
        e = np.linalg.norm(proj - img, axis=1)
        errs_chunks.append(e)
    if not errs_chunks:
        return _summarize_reprojection_errors(np.zeros(0, dtype=np.float64))
    e_all = np.concatenate(errs_chunks)
    return _summarize_reprojection_errors(e_all)


def calibrate_single_camera(cam_cfg: dict,
                            tag_yaml: str,
                            result_dir: str,
                            verbose: bool = False,
                            coverage_cfg: dict | None = None,
                            opencv_criteria: tuple | None = None):
    cam_id = cam_cfg["camera_id"]
    cam_model = cam_cfg.get("camera_model", "omni")
    img_dir = cam_cfg.get("image_path")
    sub_pix_predict = cam_cfg.get("sub_pix_predict", False)

    if img_dir is None:
        raise ValueError(f"camera_id={cam_id}: image_path is not configured.")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"camera_id={cam_id}: image directory does not exist: {img_dir}")

    # AprilTag detector
    detector = Detector(
        camera_id=cam_id,
        tag_config="tag36h11",
        minimum_tag_num=4,
        yaml_file=tag_yaml,
    )

    # read images and run tag detection
    image_files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    )

    if not image_files:
        raise RuntimeError(f"camera_id={cam_id}: no images found in directory {img_dir}")

    images = []
    shape = None
    img_idx = 0
    for fname in tqdm(image_files, desc=f"camera {cam_id} tag detection"):
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        if shape is None:
            shape = img.shape[:2]
        # always enable sub-pixel corner refinement here (consistent with TestCalib)
        _, _ = detector.detect(img, None, img_idx, show=False, enable_subpix=True)
        images.append(img)
        img_idx += 1

    if shape is None:
        raise RuntimeError(f"camera_id={cam_id}: all images failed to load, cannot calibrate.")

    h, w = shape

    result = {
        "camera_id": cam_id,
        "camera_model": cam_model,
        "image_size": [w, h],
    }

    # Save one debug image that labels 2D corner ordering (0..3) for a few tags.
    # This helps verify 2D/3D correspondences when RMSE is abnormally high.
    if verbose and detector.results:
        first_idx = sorted(detector.results.keys())[0]
        _, marker_corners, ids = detector.results[first_idx]
        debug_path = os.path.join(result_dir, f"camera_{cam_id}_corner_order_debug.png")
        ok = _save_detected_corner_order_debug(debug_path, images[first_idx], marker_corners, ids)
        if ok:
            print(f"[Debug] camera_id={cam_id}: corner order debug saved to: {debug_path}")

    # coverage-based frame selection controlled by coverage_filter in the config
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
    coverage_tracker = None

    if coverage_enable and all_indices:
        coverage_tracker = CoverageTracker(shape, grid_downsample=grid_downsample)
        used_indices = []
        for idx in all_indices:
            _, marker_corners, ids = detector.results[idx]
            poly = compute_board_polygon(marker_corners, ids)
            added_ratio, coverage_ratio, accept = coverage_tracker.update_and_check(
                poly, added_ratio_thresh
            )
            if accept:
                used_indices.append(idx)

        if not used_indices:
            # if everything is filtered out, fall back to using all frames
            used_indices = all_indices

        # save a coverage visualization image for this camera
        coverage_img_path = os.path.join(
            result_dir, f"camera_{cam_id}_coverage.png"
        )
        coverage_tracker.save_coverage_image(coverage_img_path)
        print(
            f"[Coverage] camera_id={cam_id}: coverage visualization saved to: {coverage_img_path}"
        )

    total_frames = len(all_indices)
    used_frames = len(used_indices)

    # Allow per-camera override of OpenCV termination criteria, falling back to the
    # globally provided `opencv_criteria` from the task config.
    if isinstance(cam_cfg, dict):
        oc = cam_cfg.get("opencv_criteria")
        if isinstance(oc, dict):
            max_iter = int(oc.get("max_iter", 0))
            eps = float(oc.get("eps", 0.0))
            if max_iter > 0 and eps > 0:
                opencv_criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    max_iter,
                    eps,
                )

    if cam_model == "omni":
        # use the existing omnidir mono calibrator to get an initial solution
        calibrator = IntrinsicCalibrator()
        # keep the exact data fed into OpenCV for later reprojection-error stats
        pts_3d_init, pts_2d_init = detector.gather_information(selected_indices=used_indices)
        retval_init, K, xi, D, rvecs, tvecs, idx = calibrator.calibrate_mono(
            detector, (h, w), None, None, selected_indices=used_indices
        )
        if retval_init is None:
            raise RuntimeError(f"camera_id={cam_id}: intrinsic calibration failed (insufficient points).")

        retval = float(retval_init)
        reproj_stats = _compute_reprojection_error_stats_omni(
            pts_3d_init,
            pts_2d_init,
            rvecs,
            tvecs,
            np.asarray(K, dtype=np.float64),
            float(xi[0]),
            np.asarray(D, dtype=np.float64),
        )

        # if sub_pix_predict is enabled, run two rounds of refine_calibration + omnidir.recalibration
        if sub_pix_predict:
            detection_result = detector.results
            valid_detection_result = []
            valid_detected_images = []
            used_indices_for_refine = idx[0] if hasattr(idx, "__getitem__") else idx
            for i in used_indices_for_refine:
                valid_detection_result.append(detection_result[i])
                valid_detected_images.append(images[i])

            # run two rounds of refine_calibration + cv2.omnidir.calibrate
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
                    criteria=opencv_criteria
                    or (
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        200,
                        1e-6,
                    ),
                )
                retval = float(retval_refine)
                reproj_stats = _compute_reprojection_error_stats_omni(
                    points_3d,
                    points_2d,
                    rvecs,
                    tvecs,
                    np.asarray(K, dtype=np.float64),
                    float(xi[0]),
                    np.asarray(D, dtype=np.float64),
                )

        result.update(
            {
                "rmse": float(retval),
                "rmse_init": float(retval_init),
                "K": K.tolist(),
                "xi": float(xi[0]),
                "D": D.reshape(-1).tolist(),
                "reprojection_error": reproj_stats,
            }
        )

        result["sub_pix_predict_used"] = bool(sub_pix_predict)

    elif cam_model == "pinhole":
        # standard pinhole camera model calibration
        # NOTE: reuse the coverage-filtered frame indices (used_indices) here.
        # Otherwise, when there are many frames (hundreds or more), cv2.calibrateCamera
        # may become extremely slow and look like it is "frozen".
        obj_pts, img_pts = detector.gather_information(selected_indices=used_indices)
        if not obj_pts or not img_pts:
            raise RuntimeError(f"camera_id={cam_id}: intrinsic calibration failed (insufficient points).")
        print(
            f"camera_id={cam_id}: using {len(obj_pts)}/{total_frames} frames for pinhole intrinsic "
            "calibration, calling cv2.calibrateCamera. This may take a while..."
        )
        # NOTE: OpenCV defaults to a relatively small number of iterations.
        # If you suspect "early stopping", configure cam_cfg.opencv_criteria.
        retval, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts,
            img_pts,
            (w, h),
            None,
            None,
            criteria=opencv_criteria,
        )
        reproj_stats = _compute_reprojection_error_stats_pinhole(
            obj_pts,
            img_pts,
            rvecs,
            tvecs,
            np.asarray(K, dtype=np.float64),
            np.asarray(D, dtype=np.float64),
        )
        result.update(
            {
                "rmse": float(retval),
                "K": K.tolist(),
                "D": D.reshape(-1).tolist(),
                "reprojection_error": reproj_stats,
            }
        )
        # advanced prediction + refinement pipeline is only used for omni at the moment
        result["sub_pix_predict_used"] = False
    else:
        raise ValueError(f"不支持的 camera_model: {cam_model}")

    # write coverage statistics into the result dict
    result["used_frames"] = int(used_frames)
    result["total_frames"] = int(total_frames)
    result["coverage_ratio"] = float(coverage_ratio)
    result["coverage_added_thresh"] = float(added_ratio_thresh)
    result["coverage_grid_downsample"] = int(grid_downsample)

    # save intrinsic result for this camera
    ensure_dir(result_dir)
    out_path = os.path.join(result_dir, f"camera_{cam_id}_intrinsic.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(result, f)
    print(f"[Intrinsic] camera_id={cam_id} finished. Result saved to: {out_path}")

    # Return detector and intrinsic result for downstream extrinsic calibration
    return detector, result


def _build_frame_points_from_detection(detector: Detector, frame_idx: int):
    """
    Build per-frame 3D/2D point correspondences from Detector.results entry.

    Returns:
        obj_pts: (N, 3) array of AprilGrid 3D points in board frame.
        img_pts: (N, 2) array of corresponding image points.
        If the frame does not have enough tags, returns (None, None).
    """
    if frame_idx not in detector.results:
        return None, None
    _, corners_list, ids = detector.results[frame_idx]
    if len(corners_list) < detector.minimum_tag_num:
        return None, None

    obj_pts = []
    img_pts = []
    for corners, tag_id in zip(corners_list, ids):
        # 3D corners in board frame
        grid_pts = detector.aprilgrid_3d_points[tag_id[0]].reshape(-1, 3)
        # 2D corners in image
        corners_2d = corners.reshape(-1, 2)
        obj_pts.extend(grid_pts)
        img_pts.extend(corners_2d)

    if not obj_pts or not img_pts:
        return None, None

    obj_pts = np.asarray(obj_pts, dtype=np.float32)
    img_pts = np.asarray(img_pts, dtype=np.float32)
    return obj_pts, img_pts


def _se3_increment(xi: np.ndarray) -> np.ndarray:
    """
    Construct a small SE(3) increment from a 6D vector.

    This is a simple left-multiplicative update:
        T_inc = [ R(omega)  v ]
                [    0      1 ]

    where omega (first 3) is axis-angle, v (last 3) is translation in the
    current camera frame. This is sufficient for numerical least-squares.
    """
    omega = xi[:3]
    v = xi[3:]
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        R_inc = np.eye(3, dtype=np.float64)
    else:
        R_inc, _ = cv2.Rodrigues(omega.reshape(3, 1))
    T_inc = np.eye(4, dtype=np.float64)
    T_inc[:3, :3] = R_inc
    T_inc[:3, 3] = v.reshape(3)
    return T_inc


def _rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert OpenCV (rvec, tvec) to a 4x4 homogeneous transform."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def run_extrinsic_calibration(
    config: dict,
    result_dir: str,
    detectors: dict,
    intrinsics: dict,
):
    """
    Stereo extrinsic calibration using reprojection-error based bundle adjustment.

    For each camera pair (ci, cj) in extrinsic_info.camera_pairs:
      1. Find frames where both cameras have valid tag detections.
      2. For the base camera (usually ci == base_camera_id), run PnP per frame
         to estimate board pose T_ci_board^f.
      3. For the paired camera, run PnP on at least one frame to obtain an
         initial relative transform T_cj_ci (extrinsic).
      4. Build a least-squares problem over:
           - T_cj_ci (6D increment on SE(3))
           - { T_ci_board^f } for all common frames (6D increment each)
         minimizing the pixel reprojection error in both cameras.
      5. Save the optimized extrinsic T_cj_ci to a YAML file.
    """
    extr_cfg = config.get("extrinsic_info", {})
    if not extr_cfg.get("enable", False):
        print("[Extrinsic] extrinsic_info.enable is False, skip extrinsic calibration.")
        return

    base_cam_id = int(extr_cfg.get("base_camera_id", 0))
    cam_pairs = extr_cfg.get("camera_pairs", [])
    if not cam_pairs:
        print("[Extrinsic] No camera_pairs configured, skip extrinsic calibration.")
        return

    ensure_dir(result_dir)

    for pair in cam_pairs:
        if len(pair) != 2:
            print(f"[Extrinsic] Invalid camera pair entry (expected length 2): {pair}")
            continue

        ci, cj = int(pair[0]), int(pair[1])
        if ci not in detectors or cj not in detectors:
            print(f"[Extrinsic] Missing detectors for camera pair ({ci}, {cj}), skip.")
            continue
        if ci not in intrinsics or cj not in intrinsics:
            print(f"[Extrinsic] Missing intrinsics for camera pair ({ci}, {cj}), skip.")
            continue

        intr_i = intrinsics[ci]
        intr_j = intrinsics[cj]
        if intr_i.get("camera_model") != "pinhole" or intr_j.get("camera_model") != "pinhole":
            print(
                f"[Extrinsic] Only pinhole-pinhole pairs are supported for now. "
                f"Pair ({ci}, {cj}) will be skipped."
            )
            continue

        K_i = np.asarray(intr_i["K"], dtype=np.float64)
        D_i = np.asarray(intr_i["D"], dtype=np.float64).reshape(-1, 1)
        K_j = np.asarray(intr_j["K"], dtype=np.float64)
        D_j = np.asarray(intr_j["D"], dtype=np.float64).reshape(-1, 1)

        det_i = detectors[ci]
        det_j = detectors[cj]

        # build list of common frame indices where both cameras have valid detections
        idx_i = set(det_i.results.keys())
        idx_j = set(det_j.results.keys())
        common_indices = sorted(idx_i & idx_j)
        if not common_indices:
            print(f"[Extrinsic] No common detection frames for pair ({ci}, {cj}), skip.")
            continue

        frames_data = []
        for frame_idx in common_indices:
            obj_i, img_i = _build_frame_points_from_detection(det_i, frame_idx)
            obj_j, img_j = _build_frame_points_from_detection(det_j, frame_idx)
            if obj_i is None or img_i is None or obj_j is None or img_j is None:
                continue
            if obj_i.shape[0] < 4 or obj_j.shape[0] < 4:
                continue
            frames_data.append(
                {
                    "frame_idx": frame_idx,
                    "obj_i": obj_i,
                    "img_i": img_i,
                    "obj_j": obj_j,
                    "img_j": img_j,
                    # heuristic score: use min(#points_i, #points_j) so that
                    # frames where both cameras see many tags are preferred
                    "score": min(obj_i.shape[0], obj_j.shape[0]),
                }
            )

        if not frames_data:
            print(f"[Extrinsic] No usable common frames (with enough points) for pair ({ci}, {cj}), skip.")
            continue

        # optionally subsample best frames to limit problem size
        max_frames = int(extr_cfg.get("max_frames", 80))
        if len(frames_data) > max_frames:
            # sort by score (descending), then take top max_frames
            frames_data.sort(key=lambda f: f["score"], reverse=True)
            frames_data = frames_data[:max_frames]
            # keep frames ordered by frame index for readability
            frames_data.sort(key=lambda f: f["frame_idx"])
            print(
                f"[Extrinsic] Pair ({ci}, {cj}): using top {max_frames} frames "
                f"out of {len(common_indices)} common frames based on tag coverage."
            )

        # PnP to initialize board pose for base camera (we always treat ci as the base here)
        for frame in frames_data:
            obj = frame["obj_i"].reshape(-1, 1, 3)
            img = frame["img_i"].reshape(-1, 1, 2)
            ok, rvec, tvec = cv2.solvePnP(obj, img, K_i, D_i, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                raise RuntimeError(
                    f"[Extrinsic] solvePnP failed for camera {ci}, frame {frame['frame_idx']}"
                )
            frame["T_ci_board_init"] = _rt_to_T(rvec, tvec)

        # Use the first frame with valid detections in both cameras to initialize T_cj_ci
        T_cj_ci_init = None
        for frame in frames_data:
            obj = frame["obj_j"].reshape(-1, 1, 3)
            img = frame["img_j"].reshape(-1, 1, 2)
            ok, rvec_j, tvec_j = cv2.solvePnP(obj, img, K_j, D_j, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            T_cj_board = _rt_to_T(rvec_j, tvec_j)
            T_ci_board = frame["T_ci_board_init"]
            T_cj_ci_init = T_cj_board @ np.linalg.inv(T_ci_board)
            break

        if T_cj_ci_init is None:
            print(
                f"[Extrinsic] Could not initialize T_{cj}_{ci} from PnP for pair ({ci}, {cj}), skip."
            )
            continue

        num_frames = len(frames_data)
        # parameter vector: [xi_cj_ci (6), xi_board_f0 (6), ..., xi_board_f{N-1} (6)]
        x0 = np.zeros(6 * (1 + num_frames), dtype=np.float64)

        def residual_func(x: np.ndarray) -> np.ndarray:
            xi_ex = x[:6]
            T_cj_ci = _se3_increment(xi_ex) @ T_cj_ci_init

            residuals = []
            for k, frame in enumerate(frames_data):
                xi_f = x[6 + 6 * k : 6 + 6 * (k + 1)]
                T_ci_board = _se3_increment(xi_f) @ frame["T_ci_board_init"]

                # camera i (base of this pair)
                R_ci = T_ci_board[:3, :3]
                t_ci = T_ci_board[:3, 3]
                rvec_ci, _ = cv2.Rodrigues(R_ci)
                proj_i, _ = cv2.projectPoints(
                    frame["obj_i"].reshape(-1, 1, 3),
                    rvec_ci,
                    t_ci.reshape(3, 1),
                    K_i,
                    D_i,
                )
                proj_i = proj_i.reshape(-1, 2)
                err_i = (proj_i - frame["img_i"]).reshape(-1)
                residuals.append(err_i)

                # camera j
                T_cj_board = T_cj_ci @ T_ci_board
                R_cj = T_cj_board[:3, :3]
                t_cj = T_cj_board[:3, 3]
                rvec_cj, _ = cv2.Rodrigues(R_cj)
                proj_j, _ = cv2.projectPoints(
                    frame["obj_j"].reshape(-1, 1, 3),
                    rvec_cj,
                    t_cj.reshape(3, 1),
                    K_j,
                    D_j,
                )
                proj_j = proj_j.reshape(-1, 2)
                err_j = (proj_j - frame["img_j"]).reshape(-1)
                residuals.append(err_j)

            if not residuals:
                return np.zeros(0, dtype=np.float64)
            return np.concatenate(residuals).astype(np.float64)

        print(
            f"[Extrinsic] Start optimizing pair ({ci}, {cj}) with {num_frames} common frames "
            f"and {6 * (1 + num_frames)} parameters (numeric Jacobian, method='trf')."
        )
        # Use 'trf' instead of 'lm': MINPACK/LM has a 32-bit integer limit on m*n
        # and will overflow for large problems (many frames × many points).
        res = least_squares(
            residual_func,
            x0,
            method="trf",
            verbose=1,
        )
        xi_ex_opt = res.x[:6]
        T_cj_ci_opt = _se3_increment(xi_ex_opt) @ T_cj_ci_init

        R_cj_ci = T_cj_ci_opt[:3, :3]
        t_cj_ci = T_cj_ci_opt[:3, 3]

        # compute per-point reprojection error statistics at the optimum
        r_opt = residual_func(res.x)
        if r_opt.size > 0:
            r_2d = r_opt.reshape(-1, 2)
            per_point_err = np.linalg.norm(r_2d, axis=1)
            err_mean = float(per_point_err.mean())
            err_min = float(per_point_err.min())
            err_max = float(per_point_err.max())
            num_points = int(per_point_err.size)
        else:
            err_mean = err_min = err_max = 0.0
            num_points = 0

        out_ex_path = os.path.join(result_dir, f"stereo_{ci}_{cj}_extrinsic.yaml")
        out_data = {
            "base_camera_id": int(base_cam_id),
            "pair": [int(ci), int(cj)],
            "T_cj_ci": {
                "R": R_cj_ci.tolist(),
                "t": t_cj_ci.reshape(3).tolist(),
            },
            "optimization": {
                "cost": float(res.cost),
                "num_iterations": int(res.nfev),
                "num_frames": int(num_frames),
                "reprojection_error": {
                    "mean": err_mean,
                    "min": err_min,
                    "max": err_max,
                    "num_points": num_points,
                },
            },
        }
        with open(out_ex_path, "w") as f:
            yaml.safe_dump(out_data, f)
        print(
            f"[Extrinsic] Pair ({ci}, {cj}) optimized. Result saved to: {out_ex_path} "
            f"(mean err = {err_mean:.4f} px, max err = {err_max:.4f} px, {num_points} points)"
        )


def main():
    default_cfg = os.path.join(os.path.dirname(__file__), "calibration_task.yaml")
    parser = argparse.ArgumentParser(description="Multi-camera intrinsic calibration task")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=default_cfg,
        help=f"Path to the calibration task config file (default: {default_cfg})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Save sub-pixel refinement debug images to ./debug/camera_id/",
    )
    args = parser.parse_args()

    config = load_task_config(args.config)

    tag_info = config.get("tag_info", {})
    # Currently we still use a separate AprilGrid YAML file (e.g. april_6x6.yaml).
    # If you prefer a fully config-driven setup, Detector can be extended to use tag_info only.
    tag_yaml = os.path.join(
        os.path.dirname(__file__), "april_6x6.yaml"
    )

    result_dir = config.get("result_path")
    if result_dir is None:
        result_dir = os.path.join(os.path.dirname(__file__), "results")

    cam_cfgs = config.get("camera_info", {}).get("cameras", [])
    coverage_cfg = config.get("coverage_filter", {})
    # Global OpenCV termination criteria for calibration (optional)
    # Example (top-level in YAML):
    #   opencv_criteria:
    #     max_iter: 500
    #     eps: 1e-9
    global_criteria = None
    oc = config.get("opencv_criteria")
    if isinstance(oc, dict):
        max_iter = int(oc.get("max_iter", 0))
        eps = float(oc.get("eps", 0.0))
        if max_iter > 0 and eps > 0:
            global_criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                max_iter,
                eps,
            )
    if not cam_cfgs:
        raise RuntimeError("camera_info.cameras list is empty. Please configure cameras in calibration_task.yaml.")

    # run intrinsic calibration for each camera and keep detectors/intrinsics for extrinsic
    detectors = {}
    intrinsics = {}
    for cam_cfg in cam_cfgs:
        detector, intr = calibrate_single_camera(
            cam_cfg,
            tag_yaml,
            result_dir,
            verbose=args.verbose,
            coverage_cfg=coverage_cfg,
            opencv_criteria=global_criteria,
        )
        cam_id = cam_cfg["camera_id"]
        detectors[cam_id] = detector
        intrinsics[cam_id] = intr

    # run extrinsic calibration (stereo) if enabled in config
    run_extrinsic_calibration(config, result_dir, detectors, intrinsics)


if __name__ == "__main__":
    main()


