import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Point
import cv2
import torch
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized,\
    TracedModel

from utils.datasets import letterbox

import os
from glob import glob
from ament_index_python.packages import get_package_share_directory


class ObjectDetection(Node):
    def __init__(self):
        super().__init__("ObjectDetection")
        # Parameters
        self.declare_parameter("weights", "yolov7.pt", ParameterDescriptor(description="Weights file"))
        self.declare_parameter("conf_thres", 0.25, ParameterDescriptor(description="Confidence threshold"))
        self.declare_parameter("iou_thres", 0.45, ParameterDescriptor(description="IOU threshold"))
        self.declare_parameter("device", "", ParameterDescriptor(description="Name of the device"))
        self.declare_parameter("img_size", 640, ParameterDescriptor(description="Image size"))

        weights_file = self.get_parameter("weights").get_parameter_value().string_value
        self.weights = os.path.join(get_package_share_directory('object_detection'), 'weights', weights_file)
        self.conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        self.iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.img_size = self.get_parameter("img_size").get_parameter_value().integer_value

        # Camera info and frames
        self.rgb_image = None

        self.cv_bridge = CvBridge()
        
        # Subscribers
        self.rs_sub = self.create_subscription(Image, '/image/image_raw', self.rs_callback, 10)

        #pub
        self.publisher = self.create_publisher(String, '/detect/signs', 1)  # Create a publisher for object labels

        self.signs = String()

        # Initialize YOLOv7
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device) # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size
        self.imgsz = imgsz
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1

    def rs_callback(self, msg):
        # 將ROS Image轉換成OpenCV格式
        self.rgb_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.YOLOv7_detect()

    def preProccess(self, img):
        img=torch.from_numpy(img).to(self.device)
        img=img.half() if self.half else img.float()  # uint8 to fp16/32
        img=img/255.0
        if len(img.shape)==3:
            img=img[None]
        return img

    def YOLOv7_detect(self):
        img = self.rgb_image
        im0 = img.copy()

        img=letterbox(im0, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  #BGR to RGB
        img = np.ascontiguousarray(img)
        img=self.preProccess(img)

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        t3 = time_synchronized()

        # Process detections   
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'

                    print(self.names)

                    if conf > 0.5: # Limit confidence threshold to 50% for all classes
                        # Draw a boundary box around each object
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)

                        self.signs.data = self.names[int(cls)]
                        self.publisher.publish(self.signs)

            cv2.imshow("YOLOv7 Object detection result RGB", cv2.resize(im0, None, fx=1.5, fy=1.5))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def main(args=None):
    """Run the main function."""
    rclpy.init(args=args)
    with torch.no_grad():
        node = ObjectDetection()
        rclpy.spin(node)
        rclpy.shutdown()

if __name__ == '__main__':
    main()
