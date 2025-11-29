#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
Title        : calibrator.py
Description  : <Brief description of what the script does.>
Author       : Dr. Xu Hao 
             : Peize Liu
Email        : xuhao3e8@buaa.edu.cn
             : pliuan@connect.ust.hk
Affiliation  : Institute of Unmanned Systems, Beihang University
Created Date : 2024-11-22
Last Updated : 2024-11-22

===============================================================================
Copyright (C) <Year> Dr. Xu Hao

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation; either version 2.1 of the License, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with this program; if not, see <https://www.gnu.org/licenses/lgpl-2.1.html>.

===============================================================================
"""

import os
import cv2
import numpy as np

class IntrinsicCalibrator:
    def __init__(self, calibration_method="OPENCV") -> None:
        self.calibration_method = calibration_method

    def calibrate_mono(self, detector, image_size,
                       intrinsic_init = np.array([1.24, 813, 812, 640, 360]),
                       D_init = np.array([-0.2, 0.4, 0., 0.]),
                       selected_indices=None):
        pts_3d, pts_2d = detector.gather_information(selected_indices)
        if pts_3d is None or pts_2d is None or len(pts_3d) == 0 or len(pts_2d) == 0:
            print("No enough information for calibration.")
            return None, None, None, None, None
        if self.calibration_method == "OPENCV":
            flags = cv2.omnidir.CALIB_FIX_SKEW
            if intrinsic_init is not None:
                flags |= cv2.omnidir.CALIB_USE_GUESS
            if D_init is not None:
                D_init = D_init.reshape((1, 4))
            if intrinsic_init is not None:
                K = np.array([[intrinsic_init[1], 0, intrinsic_init[3]], [0, intrinsic_init[2], intrinsic_init[4]], [0, 0, 1]], dtype=np.float64)
                xi = np.array([intrinsic_init[0]], dtype=np.float64)
            else:
                K = None
                xi = None
            rvecs = None
            tvecs = None
            retval, K, xi, D, rvecs, tvecs, idx = cv2.omnidir.calibrate(pts_3d, pts_2d, size=image_size, K=K,
                                                            xi=xi, D=D_init, flags=flags,
                                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
            print("Intrinsic calibration done: RMSE:", retval)
            print(f"Intrinsic: [{xi[0]}, {K[0, 0]}, {K[1, 1]}, {K[0, 2]}, {K[1, 2]}]")
            print("K:\n", K)
            print("D: ", D[0])

        elif self.calibration_method == "GTSAM":
            pass
        else:
            raise ValueError("The calibration method is not supported.")
        return retval, K, xi, D, rvecs, tvecs, idx

    def refine_calibration(self,
                           valid_detection_result,
                           rvecs,
                           tvecs,
                           K,
                           xi,
                           D,
                           sub_pix_window_size,
                           valid_detected_images,
                           aprilgrid_3d_points,
                           image_shape,
                           ooi_threshold=0.75,
                           debug_save_dir=None,
                           visualize=False):
        """
        使用预测 + 亚像素细化的方式，对 omni 相机标定进行二次优化。

        该实现从 TestCalib.RefineCalibration 移植而来，并去除了对全局变量的依赖。

        Args:
            valid_detection_result: list，每个元素为 (image_t, marker_corners, ids)
            rvecs, tvecs: 初始标定得到的每帧位姿
            K, xi, D: 当前内参估计
            sub_pix_window_size: 亚像素窗口大小
            valid_detected_images: 与 valid_detection_result 对应的原始图像列表
            aprilgrid_3d_points: dict[tag_id] -> (4,3) 的 3D 角点
            image_shape: (h, w)
            ooi_threshold: 判定“out of image / 黑色区域”的阈值
            debug_save_dir: 若非 None，则在此目录下存储可视化调试图
            visualize: 若 True，则使用 cv2.imshow 显示调试图

        Returns:
            points_2d, points_3d: 用于再次调用 cv2.omnidir.calibrate 的观测数据
        """
        h, w = image_shape
        points_2d = []
        points_3d = []
        image_index = 0

        if debug_save_dir is not None:
            os.makedirs(debug_save_dir, exist_ok=True)

        for det_result, r, t, img in zip(valid_detection_result, rvecs, tvecs, valid_detected_images):
            det_3d_points = []
            det_2d_points = det_result[1]
            det_idx = det_result[2]
            det_idx_tmp = []
            image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            raw_image = img.copy()

            for id_ in det_idx:
                det_3d_points.append(aprilgrid_3d_points[id_[0]])
                det_idx_tmp.append(id_[0])

            # april_tag 6x6 ; get not detected idx
            max_tag_id = max(aprilgrid_3d_points.keys()) if aprilgrid_3d_points else -1
            predict_idx = list(set(range(max_tag_id + 1)) - set(det_idx_tmp))

            det_3d_points = np.array(det_3d_points).squeeze()
            det_2d_points = np.array(det_2d_points).squeeze()
            # predict 3d
            predict_point_index = len(det_2d_points)

            for id_ in predict_idx:
                point_3d = aprilgrid_3d_points[id_]
                # project to 2d
                point_3d = point_3d.reshape(-1, 1, 3)
                point_2d, _ = cv2.omnidir.projectPoints(point_3d, r, t, K, xi.squeeze().item(), D)
                point_3d = point_3d.reshape(1, -1, 3)
                point_2d = point_2d.reshape(1, -1, 2)
                valid_2d = []
                valid_3d = []

                # strategy: remove the point outside the lens scope
                for point2, point3 in zip(point_2d, point_3d):
                    for pt, pt3d in zip(point2, point3):  # 4x2
                        if (pt[0] < 0) or (pt[0] > w) or (pt[1] < 0) or (pt[1] > h):
                            continue
                        x_min = max(int(pt[0]) - sub_pix_window_size, 0)
                        x_max = min(int(pt[0]) + sub_pix_window_size, w)
                        y_min = max(int(pt[1]) - sub_pix_window_size, 0)
                        y_max = min(int(pt[1]) + sub_pix_window_size, h)
                        region = image_gray[y_min:y_max, x_min:x_max]
                        total_pixel = region.size
                        black_pixel = np.count_nonzero(region <= 80)
                        if total_pixel == 0:
                            continue
                        if (black_pixel / total_pixel) > ooi_threshold:
                            continue
                        valid_2d.append(pt)
                        valid_3d.append(pt3d)

                    if valid_2d:
                        valid_2d_arr = np.array(valid_2d).reshape(1, -1, 2)
                        valid_3d_arr = np.array(valid_3d).reshape(1, -1, 3)
                        # concatenate 2d with existing 2d points to form at least 4 points
                        while valid_2d_arr.shape[1] < 4:
                            # copy the first point to the end
                            valid_2d_arr = np.concatenate((valid_2d_arr, valid_2d_arr[:, 0:1, :]), axis=1)
                            valid_3d_arr = np.concatenate((valid_3d_arr, valid_3d_arr[:, 0:1, :]), axis=1)
                        det_3d_points = np.concatenate((det_3d_points, valid_3d_arr), axis=0)
                        det_2d_points = np.concatenate((det_2d_points, valid_2d_arr), axis=0)

            # refine 2d points pose
            refine_2d_points = []
            for det_2d_point in det_2d_points:
                criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_COUNT, 50, 0.001)
                det_2d_point = det_2d_point.reshape(-1, 1, 2)
                corners_subpixes = cv2.cornerSubPix(
                    image_gray,
                    det_2d_point,
                    (sub_pix_window_size, sub_pix_window_size),
                    (-1, -1),
                    criteria,
                )
                refine_2d_points.append(corners_subpixes)

            # show detected points in green and predict points in red
            for index, det_point in enumerate(refine_2d_points):
                if index < predict_point_index:
                    for point in det_point:
                        cv2.circle(raw_image, (int(point[0][0]), int(point[0][1])), 5, (0, 255, 0), -1)
                else:
                    for point in det_point:
                        cv2.circle(raw_image, (int(point[0][0]), int(point[0][1])), 5, (0, 0, 255), -1)

            if debug_save_dir is not None:
                image_name = f"{image_index:06d}.jpg"
                cv2.imwrite(os.path.join(debug_save_dir, image_name), raw_image)

            if visualize:
                cv2.imshow("Image", raw_image)
                cv2.waitKey(500)

            points_2d.append(det_2d_points)
            points_3d.append(det_3d_points)
            image_index += 1

        return points_2d, points_3d

