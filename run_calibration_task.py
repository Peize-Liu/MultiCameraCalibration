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


def _blur_score_laplacian_var(image_bgr: np.ndarray) -> float:
    """
    Blur/sharpness score: variance of Laplacian on grayscale.
    Higher = sharper. Typical values depend on resolution and scene.
    """
    if image_bgr is None:
        return 0.0
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


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


def _build_colorbar_legend(height: int, max_value: float, width: int = 110) -> np.ndarray:
    """
    Build a vertical JET colorbar legend with tick labels in pixels.
    """
    h = int(height)
    w = int(width)
    max_v = float(max_value)
    if h <= 0 or w <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if max_v <= 1e-12:
        max_v = 1.0

    # Colorbar strip
    bar_w = 28
    x0 = 8
    y0 = 18
    y1 = h - 10
    bar_h = max(1, y1 - y0)

    vals = np.linspace(max_v, 0.0, bar_h, dtype=np.float32).reshape(-1, 1)
    vals_u8 = np.clip(vals / max_v * 255.0, 0.0, 255.0).astype(np.uint8)
    bar = cv2.applyColorMap(vals_u8, cv2.COLORMAP_JET)  # (bar_h,1,3)
    bar = cv2.resize(bar, (bar_w, bar_h), interpolation=cv2.INTER_NEAREST)

    legend = np.ones((h, w, 3), dtype=np.uint8) * 255
    legend[y0:y1, x0 : x0 + bar_w] = bar

    # Title
    cv2.putText(
        legend,
        "err (px)",
        (x0, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    # Ticks
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    for t in ticks:
        yy = int(y0 + (1.0 - t) * float(bar_h - 1))
        val = t * max_v
        cv2.line(legend, (x0 + bar_w + 2, yy), (x0 + bar_w + 8, yy), (0, 0, 0), 1)
        cv2.putText(
            legend,
            f"{val:.1f}",
            (x0 + bar_w + 12, yy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return legend


def _save_reproj_error_heatmap_pinhole(
    out_path: str,
    base_image: np.ndarray,
    obj_pts_list,
    img_pts_list,
    rvecs,
    tvecs,
    K: np.ndarray,
    D: np.ndarray,
    radius: int = 6,
    clip_percentile: float = 99.0,
    alpha: float = 0.45,
):
    """
    Create a reprojection-error heatmap overlay with a colorbar legend and save it.

    The heatmap is built by splatting each observed point's error onto an image-sized
    accumulator, averaging where multiple splats overlap.
    """
    if base_image is None:
        return None
    h, w = base_image.shape[:2]
    if h <= 0 or w <= 0:
        return None

    sum_map = np.zeros((h, w), dtype=np.float32)
    cnt_map = np.zeros((h, w), dtype=np.float32)
    all_err = []

    rad = max(1, int(radius))
    for obj_pts, img_pts, rvec, tvec in zip(obj_pts_list, img_pts_list, rvecs, tvecs):
        obj, img = _flatten_obj_img_points(obj_pts, img_pts)
        proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)
        e = np.linalg.norm(proj - img, axis=1).astype(np.float32)
        all_err.append(e)

        # NOTE: We must ACCUMULATE, not overwrite.
        # cv2.circle() writes constant values, so we draw into a temporary mask
        # and add err/count to accumulators.
        tmp = np.zeros((h, w), dtype=np.uint8)
        for (x, y), err in zip(img, e):
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if 0 <= xi < w and 0 <= yi < h:
                tmp[:] = 0
                cv2.circle(tmp, (xi, yi), rad, 1, -1)
                m = tmp > 0
                sum_map[m] += float(err)
                cnt_map[m] += 1.0

    if not all_err:
        return None
    all_err = np.concatenate(all_err).astype(np.float64)
    clip_p = float(clip_percentile)
    clip_p = min(100.0, max(50.0, clip_p))
    clip_val = float(np.percentile(all_err, clip_p))
    if clip_val <= 1e-6:
        clip_val = float(all_err.max()) if all_err.size else 1.0
    if clip_val <= 1e-6:
        clip_val = 1.0

    avg_map = np.zeros_like(sum_map, dtype=np.float32)
    mask = cnt_map > 1e-6
    avg_map[mask] = sum_map[mask] / cnt_map[mask]

    norm = np.clip(avg_map / float(clip_val), 0.0, 1.0)
    heat_u8 = (norm * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    base = base_image.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(base, 1.0 - float(alpha), heat_color, float(alpha), 0.0)
    # keep untouched pixels where we have no data
    overlay[~mask] = base[~mask]

    legend = _build_colorbar_legend(h, clip_val, width=110)
    canvas = np.concatenate([overlay, legend], axis=1)
    cv2.putText(
        canvas,
        f"clip @ p{int(clip_p)} = {clip_val:.2f}px",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    return {"image": os.path.basename(out_path), "clip_percentile": clip_p, "clip_value": clip_val}


def _save_worst_frames_reproj_debug_pinhole(
    out_dir: str,
    images: list,
    frame_ids: list,
    obj_pts_list,
    img_pts_list,
    rvecs,
    tvecs,
    K: np.ndarray,
    D: np.ndarray,
    worst_k: int = 8,
):
    """
    Save debug images for the worst frames (largest mean reprojection error).

    Draws detected points (green), projected points (red), and connecting lines.
    Returns a list of dicts: [{frame_id, mean, max, num_points}, ...] sorted desc by mean.
    """
    if not frame_ids:
        return []
    per_view = []
    for j, (fid, obj_pts, img_pts, rvec, tvec) in enumerate(
        zip(frame_ids, obj_pts_list, img_pts_list, rvecs, tvecs)
    ):
        obj, img = _flatten_obj_img_points(obj_pts, img_pts)
        proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)
        e = np.linalg.norm(proj - img, axis=1)
        per_view.append(
            {
                "frame_id": int(fid),
                "mean": float(e.mean()) if e.size else 0.0,
                "max": float(e.max()) if e.size else 0.0,
                "num_points": int(e.size),
            }
        )

    per_view.sort(key=lambda x: x["mean"], reverse=True)
    keep = per_view[: max(1, int(worst_k))]

    ensure_dir(out_dir)
    for item in keep:
        fid = int(item["frame_id"])
        if fid < 0 or fid >= len(images):
            continue
        img_bgr = images[fid].copy()
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        # Find index j for this frame_id in frame_ids
        try:
            j = frame_ids.index(fid)
        except ValueError:
            continue
        obj, img = _flatten_obj_img_points(obj_pts_list[j], img_pts_list[j])
        proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvecs[j], tvecs[j], K, D)
        proj = proj.reshape(-1, 2)

        for (x, y), (u, v) in zip(img, proj):
            cv2.circle(img_bgr, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)  # detected
            cv2.circle(img_bgr, (int(round(u)), int(round(v))), 3, (0, 0, 255), -1)  # projected
            cv2.line(
                img_bgr,
                (int(round(x)), int(round(y))),
                (int(round(u)), int(round(v))),
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            img_bgr,
            f"frame {fid} mean={item['mean']:.2f}px max={item['max']:.2f}px n={item['num_points']}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_bgr,
            f"frame {fid} mean={item['mean']:.2f}px max={item['max']:.2f}px n={item['num_points']}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        out_path = os.path.join(out_dir, f"{fid:06d}.png")
        cv2.imwrite(out_path, img_bgr)

    return keep


def _index_to_image_path_map(image_dir: str) -> list[str]:
    """
    Build a stable mapping from our internal frame index (img_idx) to file path.
    `calibrate_single_camera` reads images in sorted order and assigns img_idx sequentially.
    """
    files = sorted(
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    )
    return [os.path.join(image_dir, f) for f in files]


def _draw_detected_vs_projected(
    img_bgr: np.ndarray,
    detected_xy: np.ndarray,
    projected_xy: np.ndarray,
    title: str,
) -> np.ndarray:
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
    cv2.putText(
        vis,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def _save_worst_frames_stereo_debug_opencv(
    out_dir: str,
    cam0_paths: list[str],
    cam1_paths: list[str],
    frames_data: list[dict],
    K0: np.ndarray,
    D0: np.ndarray,
    K1: np.ndarray,
    D1: np.ndarray,
    R_10: np.ndarray,
    t_10: np.ndarray,
    worst_k: int = 8,
):
    """
    Save side-by-side overlays for the worst stereo frames (by mean reprojection error).

    For each frame:
      - SolvePnP on cam0 to get board pose in cam0.
      - Use (R_10, t_10) to transform board pose into cam1.
      - Project points into both cameras and compare to detections.
    """
    per_frame = []
    t_10 = t_10.reshape(3, 1)

    for f in frames_data:
        fid = int(f["frame_idx"])
        obj = f["obj"].reshape(-1, 1, 3).astype(np.float32)
        img0 = f["img_i"].reshape(-1, 1, 2).astype(np.float32)
        img1 = f["img_j"].reshape(-1, 1, 2).astype(np.float32)

        ok, rvec0, tvec0 = cv2.solvePnP(obj, img0, K0, D0, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue

        proj0, _ = cv2.projectPoints(obj, rvec0, tvec0, K0, D0)
        err0 = np.linalg.norm(proj0.reshape(-1, 2) - img0.reshape(-1, 2), axis=1)

        R0, _ = cv2.Rodrigues(rvec0)
        t0 = tvec0.reshape(3, 1)
        R1 = R_10 @ R0
        t1 = R_10 @ t0 + t_10
        rvec1, _ = cv2.Rodrigues(R1)
        proj1, _ = cv2.projectPoints(obj, rvec1, t1, K1, D1)
        err1 = np.linalg.norm(proj1.reshape(-1, 2) - img1.reshape(-1, 2), axis=1)

        e = np.concatenate([err0, err1]).astype(np.float64)
        per_frame.append(
            {
                "frame_id": fid,
                "mean": float(e.mean()) if e.size else 0.0,
                "max": float(e.max()) if e.size else 0.0,
                "num_points": int(e.size),
                "_proj0": proj0.reshape(-1, 2),
                "_proj1": proj1.reshape(-1, 2),
            }
        )

    per_frame.sort(key=lambda x: x["mean"], reverse=True)
    keep = per_frame[: max(1, int(worst_k))]

    ensure_dir(out_dir)
    out_meta = []
    for item in keep:
        fid = int(item["frame_id"])
        if fid < 0 or fid >= len(cam0_paths) or fid >= len(cam1_paths):
            continue
        im0 = cv2.imread(cam0_paths[fid])
        im1 = cv2.imread(cam1_paths[fid])
        if im0 is None or im1 is None:
            continue

        # Find corresponding frame data
        f = next((x for x in frames_data if int(x["frame_idx"]) == fid), None)
        if f is None:
            continue
        det0 = f["img_i"].reshape(-1, 2)
        det1 = f["img_j"].reshape(-1, 2)

        vis0 = _draw_detected_vs_projected(
            im0,
            det0,
            item["_proj0"],
            f"cam{int(0)} frame {fid} mean={item['mean']:.2f}px max={item['max']:.2f}px",
        )
        vis1 = _draw_detected_vs_projected(
            im1,
            det1,
            item["_proj1"],
            f"cam{int(1)} frame {fid} mean={item['mean']:.2f}px max={item['max']:.2f}px",
        )

        # pad to same height
        h0, w0 = vis0.shape[:2]
        h1, w1 = vis1.shape[:2]
        H = max(h0, h1)
        if h0 != H:
            vis0 = cv2.copyMakeBorder(vis0, 0, H - h0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if h1 != H:
            vis1 = cv2.copyMakeBorder(vis1, 0, H - h1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        combo = np.concatenate([vis0, vis1], axis=1)

        out_path = os.path.join(out_dir, f"{fid:06d}.png")
        cv2.imwrite(out_path, combo)
        out_meta.append(
            {
                "frame_id": fid,
                "mean": float(item["mean"]),
                "max": float(item["max"]),
                "num_points": int(item["num_points"]),
            }
        )

    return out_meta

def calibrate_single_camera(cam_cfg: dict,
                            tag_yaml: str,
                            result_dir: str,
                            verbose: bool = False,
                            coverage_cfg: dict | None = None,
                            opencv_criteria: tuple | None = None,
                            heatmap_cfg: dict | None = None,
                            blur_cfg: dict | None = None):
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
    blur_scores = []
    img_idx = 0
    for fname in tqdm(image_files, desc=f"camera {cam_id} tag detection"):
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        if shape is None:
            shape = img.shape[:2]
        blur_scores.append(_blur_score_laplacian_var(img))
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

    # blur/sharpness-based filtering (optional, global config blur_filter)
    blur_enable = False
    blur_thresh = 0.0
    if isinstance(blur_cfg, dict):
        blur_enable = bool(blur_cfg.get("enable", False))
        blur_thresh = float(blur_cfg.get("laplacian_var_thresh", 0.0))

    candidate_indices = all_indices
    blur_used = 0
    if blur_enable and candidate_indices:
        kept = []
        for idx in candidate_indices:
            if 0 <= idx < len(blur_scores) and blur_scores[idx] >= blur_thresh:
                kept.append(idx)
        candidate_indices = kept
        blur_used = len(candidate_indices)
        if not candidate_indices:
            # if everything is filtered out, fall back to using all frames
            candidate_indices = all_indices

    if coverage_enable and all_indices:
        coverage_tracker = CoverageTracker(shape, grid_downsample=grid_downsample)
        used_indices = []
        for idx in candidate_indices:
            _, marker_corners, ids = detector.results[idx]
            poly = compute_board_polygon(marker_corners, ids)
            added_ratio, coverage_ratio, accept = coverage_tracker.update_and_check(
                poly, added_ratio_thresh
            )
            if accept:
                used_indices.append(idx)

        if not used_indices:
            # if everything is filtered out, fall back to using all frames
            used_indices = candidate_indices

        # save a coverage visualization image for this camera
        coverage_img_path = os.path.join(
            result_dir, f"camera_{cam_id}_coverage.png"
        )
        coverage_tracker.save_coverage_image(coverage_img_path)
        print(
            f"[Coverage] camera_id={cam_id}: coverage visualization saved to: {coverage_img_path}"
        )

    elif blur_enable:
        # Only blur filter is enabled (no coverage): use candidate indices
        used_indices = candidate_indices

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
        # Map calibration views back to original frame ids (must match gather_information order)
        frame_ids_for_calib = []
        for image_idx in used_indices:
            if image_idx in detector.results:
                _, corners, _ids = detector.results[image_idx]
                if len(corners) >= detector.minimum_tag_num:
                    frame_ids_for_calib.append(int(image_idx))
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
        # Optional reprojection-error heatmap visualization
        hm_enable = True
        hm_radius = 6
        hm_clip_p = 99.0
        hm_alpha = 0.45
        hm_save_worst = False
        hm_worst_k = 8
        if isinstance(heatmap_cfg, dict):
            hm_enable = bool(heatmap_cfg.get("enable", True))
            hm_radius = int(heatmap_cfg.get("radius", hm_radius))
            hm_clip_p = float(heatmap_cfg.get("clip_percentile", hm_clip_p))
            hm_alpha = float(heatmap_cfg.get("alpha", hm_alpha))
            hm_save_worst = bool(heatmap_cfg.get("save_worst_frames", hm_save_worst))
            hm_worst_k = int(heatmap_cfg.get("worst_k", hm_worst_k))
        if hm_enable and used_indices and images:
            base_img = images[used_indices[0]] if 0 <= used_indices[0] < len(images) else images[0]
            out_img = os.path.join(result_dir, f"camera_{cam_id}_reproj_error.png")
            meta = _save_reproj_error_heatmap_pinhole(
                out_img,
                base_img,
                obj_pts,
                img_pts,
                rvecs,
                tvecs,
                np.asarray(K, dtype=np.float64),
                np.asarray(D, dtype=np.float64),
                radius=hm_radius,
                clip_percentile=hm_clip_p,
                alpha=hm_alpha,
            )
            if meta is not None:
                # record alongside numeric stats
                result["reprojection_error"].update(meta)
        if hm_save_worst and frame_ids_for_calib and images:
            out_dir = os.path.join(result_dir, f"camera_{cam_id}_reproj_worst_frames")
            worst = _save_worst_frames_reproj_debug_pinhole(
                out_dir,
                images,
                frame_ids_for_calib,
                obj_pts,
                img_pts,
                rvecs,
                tvecs,
                np.asarray(K, dtype=np.float64),
                np.asarray(D, dtype=np.float64),
                worst_k=hm_worst_k,
            )
            if worst:
                result["reprojection_error"]["worst_frames_dir"] = os.path.basename(out_dir)
                result["reprojection_error"]["worst_frames"] = worst
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

    # write blur filter stats
    if blur_enable and all_indices:
        used_blur_scores = [blur_scores[i] for i in used_indices if 0 <= i < len(blur_scores)]
        if used_blur_scores:
            result["blur_filter"] = {
                "enable": True,
                "laplacian_var_thresh": float(blur_thresh),
                "mean": float(np.mean(used_blur_scores)),
                "min": float(np.min(used_blur_scores)),
                "max": float(np.max(used_blur_scores)),
                "kept_frames": int(blur_used),
            }
        else:
            result["blur_filter"] = {
                "enable": True,
                "laplacian_var_thresh": float(blur_thresh),
                "kept_frames": int(blur_used),
            }
    else:
        result["blur_filter"] = {"enable": False}

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


def _build_common_frame_points_from_two_detections(
    det_i: Detector,
    det_j: Detector,
    frame_idx: int,
):
    """
    Build per-frame 3D/2D correspondences for stereoCalibrate by intersecting
    tag IDs detected in both cameras.

    Returns:
        obj_pts: (N,3) float32, board frame points
        img_i: (N,2) float32, image points in camera i
        img_j: (N,2) float32, image points in camera j
        If not enough common points, returns (None,None,None)
    """
    if frame_idx not in det_i.results or frame_idx not in det_j.results:
        return None, None, None
    _, corners_i, ids_i = det_i.results[frame_idx]
    _, corners_j, ids_j = det_j.results[frame_idx]
    if corners_i is None or corners_j is None or ids_i is None or ids_j is None:
        return None, None, None
    if len(corners_i) < det_i.minimum_tag_num or len(corners_j) < det_j.minimum_tag_num:
        return None, None, None

    # Map tag_id -> 2D corners (4x2)
    map_i = {int(tid[0]): c.reshape(-1, 2).astype(np.float32) for c, tid in zip(corners_i, ids_i)}
    map_j = {int(tid[0]): c.reshape(-1, 2).astype(np.float32) for c, tid in zip(corners_j, ids_j)}
    common_ids = sorted(set(map_i.keys()) & set(map_j.keys()))
    if not common_ids:
        return None, None, None

    obj = []
    img_i = []
    img_j = []
    for tid in common_ids:
        obj_tag = det_i.aprilgrid_3d_points[tid].reshape(-1, 3).astype(np.float32)
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

    method = str(extr_cfg.get("method", "opencv")).lower()
    save_worst = bool(extr_cfg.get("save_worst_frames", False))
    worst_k = int(extr_cfg.get("worst_k", 8))

    # image paths for stereo visualization (optional)
    cam_cfgs = config.get("camera_info", {}).get("cameras", [])
    cam_path_map = {}
    for cc in cam_cfgs:
        try:
            cid = int(cc.get("camera_id"))
        except Exception:
            continue
        cam_path_map[cid] = cc.get("image_path")

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
        # imageSize for OpenCV stereoCalibrate: (width, height)
        if "image_size" in intr_i:
            img_w, img_h = int(intr_i["image_size"][0]), int(intr_i["image_size"][1])
        else:
            # fallback: try from intr_j or default
            if "image_size" in intr_j:
                img_w, img_h = int(intr_j["image_size"][0]), int(intr_j["image_size"][1])
            else:
                img_w, img_h = 0, 0

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
            obj, img_i, img_j = _build_common_frame_points_from_two_detections(
                det_i, det_j, frame_idx
            )
            if obj is None:
                continue
            frames_data.append(
                {
                    "frame_idx": int(frame_idx),
                    "obj": obj,
                    "img_i": img_i,
                    "img_j": img_j,
                    "score": int(obj.shape[0]),
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

        if method == "opencv":
            # OpenCV standard stereo calibration with fixed intrinsics
            obj_pts = [f["obj"].reshape(-1, 1, 3) for f in frames_data]
            img_pts_i = [f["img_i"].reshape(-1, 1, 2) for f in frames_data]
            img_pts_j = [f["img_j"].reshape(-1, 1, 2) for f in frames_data]

            flags = cv2.CALIB_FIX_INTRINSIC
            # reuse global criteria if provided at top-level
            crit = None
            oc = config.get("opencv_criteria")
            if isinstance(oc, dict):
                max_iter = int(oc.get("max_iter", 0))
                eps = float(oc.get("eps", 0.0))
                if max_iter > 0 and eps > 0:
                    crit = (
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        max_iter,
                        eps,
                    )

            print(
                f"[Extrinsic][OpenCV] stereoCalibrate pair ({ci}, {cj}) "
                f"with {len(frames_data)} frames, FIX_INTRINSIC."
            )
            rms, _K_i, _D_i, _K_j, _D_j, R, T, E, F = cv2.stereoCalibrate(
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

            # Evaluate reprojection error per point (compute board pose per view using cam i)
            per_point_err = []
            for f in frames_data:
                obj = f["obj"].reshape(-1, 1, 3)
                img_i_v = f["img_i"].reshape(-1, 1, 2)
                ok, rvec_i, tvec_i = cv2.solvePnP(
                    obj, img_i_v, K_i, D_i, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    continue
                # project in cam i
                proj_i, _ = cv2.projectPoints(obj, rvec_i, tvec_i, K_i, D_i)
                err_i = (proj_i.reshape(-1, 2) - f["img_i"]).astype(np.float64)
                per_point_err.append(np.linalg.norm(err_i, axis=1))
                # project in cam j: board pose in cam j = (R,T) o (pose in cam i)
                R_i, _ = cv2.Rodrigues(rvec_i)
                t_i = tvec_i.reshape(3, 1)
                R_j = R @ R_i
                t_j = R @ t_i + T
                rvec_j, _ = cv2.Rodrigues(R_j)
                proj_j, _ = cv2.projectPoints(obj, rvec_j, t_j, K_j, D_j)
                err_j = (proj_j.reshape(-1, 2) - f["img_j"]).astype(np.float64)
                per_point_err.append(np.linalg.norm(err_j, axis=1))

            if per_point_err:
                e = np.concatenate(per_point_err)
                err_mean = float(e.mean())
                err_min = float(e.min())
                err_max = float(e.max())
                num_points = int(e.size)
            else:
                err_mean = err_min = err_max = 0.0
                num_points = 0

            out_ex_path = os.path.join(result_dir, f"stereo_{ci}_{cj}_extrinsic_opencv.yaml")
            out_data = {
                "base_camera_id": int(base_cam_id),
                "pair": [int(ci), int(cj)],
                "T_cj_ci": {"R": R.tolist(), "t": T.reshape(3).tolist()},
                "opencv": {"rms": float(rms), "flags": int(flags)},
                "reprojection_error": {
                    "mean": err_mean,
                    "min": err_min,
                    "max": err_max,
                    "num_points": num_points,
                },
                "num_frames": int(len(frames_data)),
            }

            # Optional worst-frames stereo overlay
            if save_worst:
                dir_i = cam_path_map.get(ci)
                dir_j = cam_path_map.get(cj)
                if isinstance(dir_i, str) and isinstance(dir_j, str) and os.path.isdir(dir_i) and os.path.isdir(dir_j):
                    paths_i = _index_to_image_path_map(dir_i)
                    paths_j = _index_to_image_path_map(dir_j)
                    out_dir = os.path.join(result_dir, f"stereo_{ci}_{cj}_worst_frames")
                    worst = _save_worst_frames_stereo_debug_opencv(
                        out_dir,
                        paths_i,
                        paths_j,
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
                f"[Extrinsic][OpenCV] Pair ({ci}, {cj}) done. Saved: {out_ex_path} "
                f"(rms={float(rms):.4f}, mean={err_mean:.4f}px, max={err_max:.4f}px)"
            )
            continue

        # === Custom BA method (legacy) ===
        # PnP to initialize board pose for base camera (we always treat ci as the base here)
        for frame in frames_data:
            obj = frame["obj"].reshape(-1, 1, 3)
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
            obj = frame["obj"].reshape(-1, 1, 3)
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
                    frame["obj"].reshape(-1, 1, 3),
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
                    frame["obj"].reshape(-1, 1, 3),
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

        out_ex_path = os.path.join(result_dir, f"stereo_{ci}_{cj}_extrinsic_custom.yaml")
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

    result_dir = config.get("result_path")
    if result_dir is None:
        result_dir = os.path.join(os.path.dirname(__file__), "results")

    # Build AprilGrid YAML from task config so the 3D board geometry matches tag_info.
    # This avoids silent mismatches (e.g., task tagSize=0.055 but april_6x6.yaml tagSize=0.022),
    # which can lead to poor calibration results.
    if isinstance(tag_info, dict) and tag_info.get("target_type", "aprilgrid") == "aprilgrid":
        ensure_dir(result_dir)
        tag_yaml = os.path.join(result_dir, "aprilgrid_from_task.yaml")
        tag_yaml_data = {
            "target_type": "aprilgrid",
            "tagCols": int(tag_info.get("tagCols", 6)),
            "tagRows": int(tag_info.get("tagRows", 6)),
            "tagSize": float(tag_info.get("tagSize", 0.055)),
            "tagSpacing": float(tag_info.get("tagSpacing", 0.3)),
        }
        with open(tag_yaml, "w") as f:
            yaml.safe_dump(tag_yaml_data, f)
    else:
        # Fallback to the repository default
        tag_yaml = os.path.join(os.path.dirname(__file__), "april_6x6.yaml")

    cam_cfgs = config.get("camera_info", {}).get("cameras", [])
    coverage_cfg = config.get("coverage_filter", {})
    heatmap_cfg = config.get("reprojection_heatmap", {})
    blur_cfg = config.get("blur_filter", {})
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
            heatmap_cfg=heatmap_cfg,
            blur_cfg=blur_cfg,
        )
        cam_id = cam_cfg["camera_id"]
        detectors[cam_id] = detector
        intrinsics[cam_id] = intr

    # run extrinsic calibration (stereo) if enabled in config
    run_extrinsic_calibration(config, result_dir, detectors, intrinsics)


if __name__ == "__main__":
    main()


