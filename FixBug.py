import os
import cv2
import numpy as np
from AprilDetection.detection import Detector
from Calibrator.calibrator import IntrinsicCalibrator
from AprilDetection.aprilgrid import generate_aprilgrid_3d_points, load_aprilgrid_config


if __name__ == "__main__":
    file_path = "/home/dji/workspace/QuadCalib/data/CAM_A"
    april_tag_yaml = "/home/dji/workspace/QuadCalib/april_6x6.yaml"
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
    ooi_area = 4 # out of image area; search around the predicted point and check if the surrounding pixels are black
    ooi_threadhold = 0.5 # out of image threadhold
    image = "/home/dji/workspace/QuadCalib/expriment_data/000105.jpg"
    img = cv2.imread(image)
    #debug
    f = open("/home/dji/workspace/QuadCalib/debug.txt", "w")
    raw_image_dir = "/home/dji/workspace/QuadCalib/data/raw_data"
    detect_image_dir = "/home/dji/workspace/QuadCalib/data/detected_data"

    r = np.array([[-2.44974159],
                [ 0.06784736],
                [ 1.93859513]], dtype=np.float64)
    t = np.array([[0.61319483],
                    [0.46540802],
                    [0.47306995]], dtype=np.float64)

    K = np.array([[959.38504762, 0.0,  671.57355472],
                [0.0, 959.68437352, 364.16891744],
                [0.0, 0.0, 1.0]], dtype=np.float64)
    xi = np.array([[1.60765862]], dtype=np.float64)
    D = np.array([-0.33645712, 0.16370223,  0.00080834, -0.00049488])

    debug_detector = Detector(camera_id=0, tag_config="tag36h11", minimum_tag_num=4, yaml_file=april_tag_yaml)
    show_img, det_result = debug_detector.detect(img, None, image_num, show=True, enable_subpix=True)
    det_result = debug_detector.results

    # cv2.imshow("Image", img)
    # cv2.waitKey(500)
    pred_2d_points = []
    pred_3d_points = []
    det_3d_points = []
    det_2d_points = det_result[0][1]
    det_idx = det_result[0][2]
    det_idx_tmp = []
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raw_image = img.copy()
    for id in det_idx:
        det_3d_points.append(aprilgrid_3d_points[id[0]])
        det_idx_tmp.append(id[0])
    # april_tag 6x6 ; get not detected idx
    shape = image_gray.shape
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
        for point2,point3 in zip(point_2d,point_3d) :
            for pt, pt3d in zip(point2, point3):
                if (pt[0] < 0) or (pt[0] > shape[1]) or (pt[1] < 0) or (pt[1] > shape[0]):
                    continue
                x_min = max(int(pt[0]) - ooi_area, 0)
                x_max = min(int(pt[0]) + ooi_area, shape[1])
                y_min = max(int(pt[1]) - ooi_area, 0)
                y_max = min(int(pt[1]) + ooi_area, shape[0])
                region = image_gray[y_min:y_max, x_min:x_max]
                total_pixel = region.size
                black_pixel = np.count_nonzero(region <= 60)
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
        corners_subpixes = cv2.cornerSubPix(image_gray, det_2d_point, (3,3), (-1,-1), criteria)
        refine_2d_points.append(corners_subpixes)
    # show detected points in green and predict points in red
    for index, det_point in enumerate(refine_2d_points):
        if (index < predict_point_index):
            for point in det_point:
                cv2.circle(img, (int(point[0][0]), int(point[0][1])), 5, (0, 255, 0), -1)
        else:
            for point in det_point:
                cv2.circle(img, (int(point[0][0]), int(point[0][1])), 5, (0, 0, 255), -1)
    
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    cv2.imwrite("test.jpg", img)
    # save image
    # image_name = format(image_index, '06d')
    # image_name = image_name + ".jpg"
    # cv2.imwrite(os.path.join(detect_image_dir, image_name), img)
    # cv2.imwrite(os.path.join(raw_image_dir, image_name), raw_image)
    # #debug save image; save gray image and detect-image; r t
    # f.write(f"{os.path.join(detect_image_dir, image_name)} ; {os.path.join(raw_image_dir, image_name)} ; {r}  ; {t}\n", )
    # points_2d.append(det_2d_points)
    # points_3d.append(det_3d_points)
    # image_index = image_index + 1
        
    
    # retval, K, xi, D, rvecs, tvecs, idx = cv2.omnidir.calibrate(points_3d, points_2d, shape, K, xi, D, flags=cv2.omnidir.CALIB_USE_GUESS, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    # print("Intrinsic calibration done: RMSE:", retval)
    # print(f"Intrinsic: [{xi[0]}, {K[0, 0]}, {K[1, 1]}, {K[0, 2]}, {K[1, 2]}]")
    # print("K:\n", K)
    # print("D: ", D[0])