import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.br = CvBridge()

    def listener_callback(self, img_msg):

        cv_img = self.br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        cv.imshow("Matches", cv_img)

        if cv.waitKey(10) & 0xFF == ord('q'):
            self.get_logger().info("Closing subscriber")
            rclpy.shutdown()

        self.get_logger().info('I heard...')



def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
