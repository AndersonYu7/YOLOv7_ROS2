# Object Detection -- YOLOv7

## Prerequisites


## Execution
### Step 1: Open Camera

* [Camera](README.md#Use-H65-Camera)

### Step 2: Launch the detect program 
`$ ros2 launch object_detection object_detection_launch.py `

#### Parameter Description
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

## Model File Location

The YOLOv7 model weights file should be placed in the following location:

- `detect/object_detection/weights`: Store the YOLOv7 model weights file in this directory.

## References
* [YOLOv7](https://github.com/WongKinYiu/yolov7.git)
* [YOLOv7 ROS2](https://github.com/Marnonel6/YOLOv7_ROS2.git)
