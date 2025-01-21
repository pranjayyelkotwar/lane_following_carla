#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import FastSAM

class RoadBinaryMasker:
    def __init__(self, model_path='FastSAM-s.pt', device='cuda'):
        self.model = FastSAM(model_path).to(device)

    def get_mask(self, image):
        results = self.model.predict(image)  
        masks = results[0].masks.data
        masks = masks.cpu().numpy()   
        h = masks.shape[1]
        w = masks.shape[2]

        road_points = [h - h // 5, w // 2]
        road_point_y, road_point_x = road_points
        road_mask_index = None
        for i in range(len(masks)):
            if masks[i][road_point_y][road_point_x] == 1:
                road_mask_index = i
                break

        if road_mask_index is None:
            return None
        
        road_mask = masks[road_mask_index]
        return road_mask

class SteeringAngleCalculator:
    def __init__(self, img_width, img_height, clip_range=(20, 60)):
        self.img_width = img_width
        self.img_height = img_height
        self.clip_range = clip_range

    def calculate_steering_angle(self, road_mask, debug=False):
        clip_start = int(self.img_height * (self.clip_range[0] / 100))
        clip_end = int(self.img_height * (self.clip_range[1] / 100))
        clipped_mask = road_mask[clip_start:clip_end, :]

        edges = cv2.Canny((clipped_mask * 255).astype(np.uint8), 50, 150)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

        visualization_img = cv2.cvtColor((road_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        if lines is None:
            return 0.0, visualization_img

        left_lines = []
        right_lines = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                y1 += clip_start
                y2 += clip_start
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if slope < 0 and x1 < self.img_width // 2:
                    left_lines.append((x1, y1, x2, y2))
                    cv2.line(visualization_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                elif slope > 0 and x2 > self.img_width // 2:
                    right_lines.append((x1, y1, x2, y2))
                    cv2.line(visualization_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if not left_lines and not right_lines:
            return 0.0, visualization_img

        def average_slope_intercept(lines):
            x_coords, y_coords = [], []
            for x1, y1, x2, y2 in lines:
                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])
            poly = np.polyfit(x_coords, y_coords, 1)
            slope = poly[0]
            intercept = poly[1]
            return slope, intercept

        left_slope, left_intercept = average_slope_intercept(left_lines) if left_lines else (None, None)
        right_slope, right_intercept = average_slope_intercept(right_lines) if right_lines else (None, None)

        if left_slope is not None and right_slope is not None:
            intersection_x = (right_intercept - left_intercept) / (left_slope - right_slope)
            intersection_y = left_slope * intersection_x + left_intercept
        elif left_slope is not None:
            intersection_x = self.img_width
            intersection_y = left_slope * intersection_x + left_intercept
        elif right_slope is not None:
            intersection_x = 0
            intersection_y = right_slope * intersection_x + right_intercept
        else:
            return 0.0, visualization_img

        intersection_point = (int(intersection_x), int(intersection_y))
        cv2.circle(visualization_img, intersection_point, 5, (255, 0, 0), -1)

        vehicle_mid_x = self.img_width // 2
        vehicle_mid_y = self.img_height
        delta_x = intersection_x - vehicle_mid_x
        delta_y = vehicle_mid_y - intersection_y

        if delta_y == 0:
            return 0.0, visualization_img

        steering_angle = np.arctan(delta_x / delta_y) * 180 / np.pi

        cv2.line(visualization_img, (vehicle_mid_x, vehicle_mid_y), intersection_point, (255, 255, 0), 2)
        return steering_angle, visualization_img

class CarlaImageViewer(Node):
    def __init__(self):
        super().__init__('carla_image_viewer')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/carla/hero/rgb_front/image',
            self.image_callback,
            10
        )
        self.masker = RoadBinaryMasker(device='cuda')
        self.calculator = None

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width, _ = cv_image.shape

            if self.calculator is None:
                self.calculator = SteeringAngleCalculator(width, height, clip_range=(20, 60))

            # Get the road mask
            road_mask = self.masker.get_mask(cv_image)
            if road_mask is None:
                self.get_logger().warn("No road detected in the image.")
                return

            # Calculate steering angle and get visualization
            steering_angle, visualization = self.calculator.calculate_steering_angle(road_mask, debug=True)
            self.get_logger().info(f"Calculated Steering Angle: {steering_angle:.2f} degrees")

            # Display the visualization
            cv2.imshow('CARLA RGB Front Camera with Steering Lines', visualization)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    carla_image_viewer = CarlaImageViewer()
    rclpy.spin(carla_image_viewer)
    carla_image_viewer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()