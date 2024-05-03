# Object Detection -- YOLOv7
![ROS 2](https://img.shields.io/badge/ROS2-humble-blue.svg)
![Pytorch](https://img.shields.io/badge/Pytorch-blue.svg)
![YOLOv7](https://img.shields.io/badge/YOLOv7-green.svg)

This package performs object detection using the **YOLOv7** model with **ROS 2**. It subscribes to camera image data and publishes detection results on a specified ROS 2 topic. The node can be used for real-time object detection in a ROS 2 environment.

<div align=center>
<img src="https://github.com/AndersonYu7/YOLOv7_ROS2/assets/95768254/2a480fce-92e8-466f-83db-b9f9766056dd" width="640" height="480">
</div>

## Prerequisites
- **ROS 2 Humble**: Ensure that your environment is set up with ROS 2 Humble.
- **PyTorch**: Deep learning framework required for YOLOv7. Install the appropriate version based on your environment (e.g., CPU or CUDA).
- **Python Packages**: Additional Python packages needed for various functionalities:
  - `tqdm`
  - `pandas`
  - `requests`
  - `seaborn`
 
This node has been tested with the following software configuration:

- **Operating System**: Ubuntu 22.04
- **Python**: 3.10.12
- **ROS 2**: Humble
- **CUDA**: 11.8

- **Python packages:**:
  - **NumPy**: 1.24.4
  - **OpenCV-Python**: 4.9.0.80
  - **PyTorch**: 2.2.2+cu118
  - **Torchvision**: 0.17.2+cu118
  - **Torchaudio**: 2.2.2+cu118
  - **tqdm**: 4.66.2
  - **pandas**: 2.2.2
  - **requests**: 2.31.0
  - **seaborn**: 0.13.2

## Usage
### Step 1: Install the ROS 2 Package
- Clone the Git repository to your workspace

`$ git clone https://github.com/AndersonYu7/YOLOv7_ROS2.git detect`

### Step 2: **Ensure the Camera is Running**
- Make sure your camera is running
- Ensure it's publishing image data to the `/image/image_raw` topic.

### Step 3: Launch the detect program 
`$ ros2 launch yolov7_obj_detect object_detection_launch.py `

## Detection results

This node publishes detection results on the `/detect/objs` topic. Each message contains the following information:

- **Labels**: The names of the detected objects.
- **Scores**: The confidence scores for each detection.
- **Bounding Boxes**: The coordinates of the bounding boxes for each detected object, represented as `(xmin, ymin, xmax, ymax)`.


## Parameter Description
* **weights**:
  - Provide the name of the YOLOv7 model weights file

* **conf_thres**: Confidence threshold for object detection.
  - Set the confidence threshold (conf_thres) to a value between 0 and 1.

* **iou_thres**: Intersection over Union (IOU) threshold for object detection.
  - Set the IOU threshold (iou_thres) to a value between 0 and 1.

* **device**: Specify the device for object detection. This parameter allows inputs like "cpu" or "0,1,2,3". By default, the object detection algorithm uses CUDA on GPUs if available, otherwise it uses CPU.
  - Provide the device (device) where the object detection algorithm will be executed. Use "cpu" for CPU or specify GPU device IDs like "0,1,2,3".

* **img_size**: Image size for object detection.
  - Set the image size (img_size) to a suitable value, such as 640, depending on the requirements of the object detection model.

* **show_img**:
  - Whether to display the detection results.

## Model File Location

The YOLOv7 model weights file should be placed in the following location:

- `detect/yolov7_obj_detect/weights`: Store the YOLOv7 model weights file in this directory.

## Notes
Ensure that your camera is properly configured and running. If you're not receiving images, check the camera connection and topic name.

## References
* [YOLOv7](https://github.com/WongKinYiu/yolov7.git)
* [YOLOv7 ROS2](https://github.com/Marnonel6/YOLOv7_ROS2.git)
