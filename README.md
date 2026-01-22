# Intro

# Usage

```
pip3 install -r requirements.txt
```

## ROS2 bag workflow (new)

1) Copy and edit `calibration_bag_task.yaml`:
- Set `bag_info.bag_path`
- Set each camera's `image_topic` (and `image_type`: `raw` or `compressed`)
- Set `imu_info.imu_topic`

2) Run:

```
python3 run_calibration_bag_task.py -c calibration_bag_task.yaml
```

Outputs are written into `result_path`:
- `camera_<id>_intrinsic.yaml`
- `imu_extrinsic.yaml` (rotation + dt)