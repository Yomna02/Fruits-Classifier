import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial

class SerialNode(Node):
    def __init__(self):
        super().__init__('serial_node')
        self.subscription = self.create_subscription(String,'result', self.listener_callback, 10)
        self.subscription 
        self.ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
        self.ser.flush()

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
        self.ser.write(f"{msg.data}\n".encode('utf-8'))

def main(args=None):
    rclpy.init(args=args)
    serial_node = SerialNode()
    rclpy.spin(serial_node)
    serial_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()