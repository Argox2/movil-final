import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image


class MinimalPublisher(Node):

    def __init__(self, cam):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Image, 'topic', 10)
        timer_period = 1/30  # 30Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.cam = cam
        self.br = CvBridge()

    def timer_callback(self):

        status, photo = self.cam.read()
        if not status:
            self.get_logger().error("Error: Could not read frame from webcam.")
            return
        
        img_msg = self.br.cv2_to_imgmsg(photo, encoding="bgr8")

        self.publisher_.publish(img_msg)
        self.get_logger().info('Publishing image...')

    def destroy_node(self):
        self.cam.release()
        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)
    cam = cv.VideoCapture(0)

    minimal_publisher = MinimalPublisher(cam)
    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
