import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

class FruitPredict(Node):
    def __init__(self):
        super().__init__('fruit_predict')
        self.publisher_ = self.create_publisher(String, 'result', 10)
        self.model = load_model('/home/youmna/Documents/Detecto Project/Model_ResNet50/model_resnet50.keras')
        self.timer = self.create_timer(1, self.classify_image)

    def capture_image(self, filename='captured_image.jpg'):
        #Initialize the webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.get_logger().info("Cannot open camera")
            return None

        #Capture a single frame
        ret, frame = cap.read()
        if not ret:
            self.get_logger().info("Can't receive frame (stream end?). Exiting ...")
            return None

        #Save captured image
        cv2.imwrite(filename, frame)
        cap.release()
        return filename

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array_expanded_dims)

    def classify_image(self):
        #Capture an image
        img_path = self.capture_image()
        if img_path is None:
            self.get_logger().info('Failed to capture image.')
            return

        self.get_logger().info(f'Processing image: {img_path}')
        processed_img = self.preprocess_image(img_path)

        #Predict
        prediction = self.model.predict(processed_img)
        class_idx = np.argmax(prediction, axis=1)[0]

        #Class 0 is apple and class 1 is banana
        result = 'apple' if class_idx == 0 else 'banana'

        self.get_logger().info(f'Predicted class: {result}')
        self.publisher_.publish(String(data=result))

def main(args=None):
    rclpy.init(args=args)
    fruit_predict = FruitPredict()
    rclpy.spin(fruit_predict)
    fruit_predict.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
