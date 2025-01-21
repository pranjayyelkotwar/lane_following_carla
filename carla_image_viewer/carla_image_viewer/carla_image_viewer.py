#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CarlaImageViewer(Node):
    def __init__(self):
        super().__init__('carla_image_viewer')

        # Create a subscriber to the RGB image topic
        self.subscription = self.create_subscription(
            Image,
            '/carla/hero/rgb_front/image',
            self.image_callback,
            10  # Queue size
        )
        self.subscription  # Prevent unused variable warning

        # Initialize CV Bridge
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            # Convert the ROS image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Display the image in an OpenCV window
            cv2.imshow('CARLA RGB Front Camera', cv_image)
            cv2.waitKey(1)  # Wait for a short period to update the window

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)

    # Create the node
    carla_image_viewer = CarlaImageViewer()

    # Spin the node to keep it alive
    rclpy.spin(carla_image_viewer)

    # Destroy the node and shutdown ROS 2
    carla_image_viewer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()