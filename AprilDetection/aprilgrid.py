import yaml
import numpy as np

def load_aprilgrid_config(yaml_file):
    try:
        """加载 AprilGrid 配置文件"""
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        
        target_type = config.get('target_type', 'aprilgrid')
        if target_type != 'aprilgrid':
            raise ValueError(f"Unsupported target type: {target_type}")
    except Exception as e:
        print(f"Error: {e}, will load default configuration.")
        config = {
            'tagCols': 6,
            'tagRows': 6,
            'tagSize': 0.088,
            'tagSpacing': 0.3
        }    
    return config

def generate_aprilgrid_3d_points(tagCols, tagRows, tagSize, tagSpacing):
    """
    根据配置生成 AprilGrid 的 3D 点，并以 {id: [[pt1], [pt2], [pt3], [pt4]]} 的形式返回
    """
    grid_points = {}
    tag_stride = tagSize * (1 + tagSpacing)  # 标签中心之间的间距
    for row in range(tagRows):
        for col in range(tagCols):
            # 计算当前标签的 4 个角点坐标
            # IMPORTANT: Keep corner ordering consistent with our OpenCV detection output.
            #
            # Verified on `camera_*_corner_order_debug.png` (see run_calibration_task.py):
            # detected 2D corner indices are:
            #   0 = bottom-right, 1 = bottom-left, 2 = top-left, 3 = top-right
            # We generate 3D corners in the SAME order so that (3D[i] <-> 2D[i]) matches.
            #
            # ALSO IMPORTANT: Tag ID layout (verified / required by your board):
            #   - bottom-left tag is ID 0
            #   - IDs increase left->right within a row
            #   - then continue on the next row UP (bottom-to-top)
            # In this implementation, `row` increases downward, so we map:
            #   tag_id = (tagRows - 1 - row) * tagCols + col
            tag_id = (tagRows - 1 - row) * tagCols + col
            tag_origin_x = col * tag_stride
            tag_origin_y = row * tag_stride
            # Here we use a simple grid coordinate where x increases to the right and
            # y increases as `row` increases (i.e., downward on the printed board layout).
            # So the tag corners are:
            #   tl = (x,   y)
            #   tr = (x+S, y)
            #   bl = (x,   y+S)
            #   br = (x+S, y+S)
            tl = [tag_origin_x, tag_origin_y, 0]
            tr = [tag_origin_x + tagSize, tag_origin_y, 0]
            bl = [tag_origin_x, tag_origin_y + tagSize, 0]
            br = [tag_origin_x + tagSize, tag_origin_y + tagSize, 0]
            # Match detected order: [br, bl, tl, tr]
            tag_corners = [br, bl, tl, tr]
            grid_points[tag_id] = np.array(tag_corners, dtype=np.float32)
    
    return grid_points

