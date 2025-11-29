import os
import cv2
import numpy as np
from AprilDetection.detection import Detector
from Calibrator.calibrator import IntrinsicCalibrator
from AprilDetection.aprilgrid import generate_aprilgrid_3d_points, load_aprilgrid_config

def RefineCalibration(valid_detection_result, rvecs, tvecs, K, xi, D, sub_pix_window_size, valid_detected_images, visualize=False):
    image_index = 0
    for det_result, r, t, img in zip(valid_detection_result, rvecs, tvecs, valid_detected_images):
    # cv2.imshow("Image", img)
    # cv2.waitKey(500)
        det_3d_points = []
        det_2d_points = det_result[1]
        det_idx = det_result[2]
        det_idx_tmp = []
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw_image = img.copy()
        for id in det_idx:
            det_3d_points.append(aprilgrid_3d_points[id[0]])
            det_idx_tmp.append(id[0])
        # april_tag 6x6 ; get not detected idx
        predict_idx = list(set(range(36)) - set(det_idx_tmp))
        
        det_3d_points = np.array(det_3d_points).squeeze()
        det_2d_points = np.array(det_2d_points).squeeze()
        # predict 3d
        predict_point_index = len(det_2d_points)
        for id in predict_idx:
            point_3d = aprilgrid_3d_points[id]
            #project to 2d
            point_3d = point_3d.reshape(-1, 1, 3)
            point_2d, _ = cv2.omnidir.projectPoints(point_3d, r, t, K, xi.squeeze().item(), D)
            point_3d = point_3d.reshape(1, -1, 3)
            point_2d = point_2d.reshape(1, -1, 2)
            valid_2d = []
            valid_3d = []
            #check if point2d_window is in the image; check point2d surrounding pixels are 80% black; if yes; pixel should not be detected
            # if not, add to detected points
            if 0:
                for point2,point3 in zip(point_2d,point_3d) :
                    if (point2[0][0] < 0) or (point2[0][0] > shape[1]) or (point2[0][1] < 0) or (point2[0][1] > shape[0]):
                        continue
                    x_min = max(int(point2[0][0]) - sub_pix_window_size, 0)
                    x_max = min(int(point2[0][0]) + sub_pix_window_size, shape[1])
                    y_min = max(int(point2[0][1]) - sub_pix_window_size, 0)
                    y_max = min(int(point2[0][1]) + sub_pix_window_size, shape[0])
                    region = image_gray[y_min:y_max, x_min:x_max]
                    total_pixel = region.size
                    black_pixel = np.count_nonzero(region <= 50)
                    if (black_pixel/total_pixel) > ooi_threadhold:
                        continue
                    valid_2d.append(point2)
                    valid_3d.append(point3)
                det_3d_points = np.concatenate((det_3d_points, valid_3d), axis=0)
                det_2d_points = np.concatenate((det_2d_points, valid_2d), axis=0)
            else: # this strategy is to remove the point outside the lens scope
                for point2,point3 in zip(point_2d,point_3d) :
                    for pt, pt3d in zip(point2, point3): # 4x2
                        if (pt[0] < 0) or (pt[0] > shape[1]) or (pt[1] < 0) or (pt[1] > shape[0]):
                            continue
                        x_min = max(int(pt[0]) - sub_pix_window_size, 0)
                        x_max = min(int(pt[0]) + sub_pix_window_size, shape[1])
                        y_min = max(int(pt[1]) - sub_pix_window_size, 0)
                        y_max = min(int(pt[1]) + sub_pix_window_size, shape[0])
                        region = image_gray[y_min:y_max, x_min:x_max]
                        total_pixel = region.size
                        black_pixel = np.count_nonzero(region <= 80)
                        if (black_pixel/total_pixel) > ooi_threadhold:
                            continue
                        valid_2d.append(pt)
                        valid_3d.append(pt3d)
                
                    valid_2d = np.array(valid_2d).reshape(1, -1, 2)
                    valid_3d = np.array(valid_3d).reshape(1, -1, 3)
                    #concatenate 2d with exsiting 2d points to form a 4 points
                    while valid_2d.shape[1] < 4:
                        #copy the first point to the end
                        valid_2d = np.concatenate((valid_2d, valid_2d[:, 0:1, :]), axis=1)
                        valid_3d = np.concatenate((valid_3d, valid_3d[:, 0:1, :]), axis=1)
                det_3d_points = np.concatenate((det_3d_points, valid_3d), axis=0)
                det_2d_points = np.concatenate((det_2d_points, valid_2d), axis=0)
        # refine 2d points pose
        refine_2d_points = []
        for det_2d_point in det_2d_points:
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_COUNT, 50, 0.001 )
            det_2d_point = det_2d_point.reshape(-1, 1, 2)
            corners_subpixes = cv2.cornerSubPix(image_gray, det_2d_point, (sub_pix_window_size,sub_pix_window_size), (-1,-1), criteria)
            refine_2d_points.append(corners_subpixes)
        # show detected points in green and predict points in red
        for index, det_point in enumerate(refine_2d_points):
            if (index < predict_point_index):
                for point in det_point:
                    cv2.circle(raw_image, (int(point[0][0]), int(point[0][1])), 5, (0, 255, 0), -1)
            else:
                for point in det_point:
                    cv2.circle(raw_image, (int(point[0][0]), int(point[0][1])), 5, (0, 0, 255), -1)
        
        if visualize:
            cv2.imshow("Image", raw_image)
            cv2.waitKey(500)
        # save image
        image_name = format(image_index, '06d')
        image_name = image_name + ".jpg"
        cv2.imwrite(os.path.join(detect_image_dir, image_name), raw_image)
        # cv2.imwrite(os.path.join(raw_image_dir, image_name), raw_image)
        #debug save image; save gray image and detect-image; r t
        # f.write(f"{os.path.join(detect_image_dir, image_name)} ; {os.path.join(raw_image_dir, image_name)} ; {r}  ; {t}\n", )
        points_2d.append(det_2d_points)
        points_3d.append(det_3d_points)
        image_index = image_index + 1
    return points_2d, points_3d


if __name__ == "__main__":
    file_path = "/home/nvidia/workspace/MultiCameraCalibration/20251128_173034_582/cam0"
    april_tag_yaml = "/home/nvidia/workspace/MultiCameraCalibration/april_6x6.yaml"
    detector = Detector(camera_id=0, tag_config="tag36h11", minimum_tag_num=4, yaml_file=april_tag_yaml)
    images = os.listdir(file_path)
    
    config = load_aprilgrid_config(april_tag_yaml)
    aprilgrid_3d_points = generate_aprilgrid_3d_points(config['tagCols'],
                                                        config['tagRows'],
                                                        config['tagSize'],
                                                        config['tagSpacing'])

    show = False
    shape = None
    image_num = 0
    image_list = []
    ooi_area = 6 # out of image area; search around the predicted point and check if the surrounding pixels are black
    ooi_threadhold = 0.75 # out of image threadhold
    for image in images:
        image_path = os.path.join(file_path, image)
        if shape is None:
            shape = cv2.imread(image_path).shape[:2]
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        show_img, valid_corner_detection = detector.detect(img, None, image_num, show=True, enable_subpix=True)
        if ((show_img is not None) and (show == True)):
            cv2.imshow("Image", show_img)
            cv2.waitKey(500)
        elif valid_corner_detection is None:
            print("[Error] No enough information for calibration.")
            continue
        image_list.append(img)
        image_num += 1

    calibrator = IntrinsicCalibrator()
    retval, K , xi, D, rvecs, tvecs, idx = calibrator.calibrate_mono(detector,shape, None, None)
    # get detection result
    detection_result = detector.results
    init_K = K
    init_D = D
    valid_detection_result = []
    for i in idx[0]:
        valid_detection_result.append(detection_result[i])

    valid_detected_images = []
    for i in idx[0]:
        valid_detected_images.append(image_list[i])

    # check detect result
    show = False
    if (show == True):
        for img, detection_result, r ,t in zip(valid_detected_images, valid_detection_result, rvecs, tvecs):
            # print("showing detection result")
            corners = detection_result[1]
            idx = detection_result[2].squeeze()
            for corner, id in zip(corners,idx):
                corner = corner.squeeze()
                center = corner.mean(axis=0)
                for dot in corner:
                    cv2.circle(img, (int(dot[0]), int(dot[1])), 5, (0, 255, 0), -1)
                cv2.putText(img, str(id), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Image", img)
            cv2.waitKey(500)
    
    # gtsam calibrate
    print("valid_img {}", len(valid_detected_images))
    print("valid_detection {}", len(valid_detection_result))

    # recalibrate
    points_2d = []
    points_3d = []
    
    #debug
    f = open("/home/nvidia/workspace/MultiCameraCalibration/debug.txt", "w")
    
    detect_image_dir = "/home/nvidia/workspace/MultiCameraCalibration/detected_data_remove_ooi_all_5x5_2_seq_iter_1"
    if not os.path.exists(detect_image_dir):
        os.makedirs(detect_image_dir)
    points_2d, points_3d = RefineCalibration(valid_detection_result, rvecs, tvecs, K, xi, D, 5, valid_detected_images)
    retval, K, xi, D, _, _, idx = cv2.omnidir.calibrate(points_3d, points_2d, shape, K, xi, D, flags=cv2.omnidir.CALIB_USE_GUESS, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6))
    
    print("[1] Intrinsic calibration done: RMSE:", retval)
    print(f"Intrinsic: [{xi[0]}, {K[0, 0]}, {K[1, 1]}, {K[0, 2]}, {K[1, 2]}]")
    print("K:\n", K)
    print("D: ", D[0])
    points_2d= []
    points_3d = []

    detect_image_dir = "/home/nvidia/workspace/MultiCameraCalibration/detected_data_remove_ooi_all_5x5_2_seq_iter_2"
    if not os.path.exists(detect_image_dir):
        os.makedirs(detect_image_dir)
    points_2d, points_3d = RefineCalibration(valid_detection_result, rvecs, tvecs, K, xi, D, 5, valid_detected_images)
    retval, K, xi, D, rvecs, tvecs, idx = cv2.omnidir.calibrate(points_3d, points_2d, shape, K, xi, D, flags=cv2.omnidir.CALIB_USE_GUESS, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6))
    
    print("[2] Intrinsic calibration done: RMSE:", retval)
    print(f"Intrinsic: [{xi[0]}, {K[0, 0]}, {K[1, 1]}, {K[0, 2]}, {K[1, 2]}]")
    print("K:\n", K)
    print("D: ", D[0])