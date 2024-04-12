import sys
import cv2
import json
import pika
import mediapipe as mp
from datetime import datetime
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from utils import PoseProcessor
from qtawesome import icon

from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QComboBox,
    QPushButton,
    QFileDialog,
    QHBoxLayout, 
    QSizePolicy
)



class MessageReceiverThread(QThread):
    message_received = pyqtSignal(str)

    def __init__(self, channel, queue_name):
        super(MessageReceiverThread, self).__init__()
        self.channel = channel
        self.queue_name = queue_name

    def run(self):
        def callback(ch, method, properties, body):
            message = json.loads(body.decode('utf-8'))
            exercise_class = message.get('class')
            self.message_received.emit(exercise_class)

        self.channel.basic_consume(queue=self.queue_name, on_message_callback=callback, auto_ack=True)
        self.channel.start_consuming()


class RabbitMQManager:
    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='hello')
        self.channel.queue_declare(queue='prediction')


class VideoWindow(QWidget):
    def __init__(self, rabbitmq_manager):
        super().__init__()
        self.rabbitmq_manager = rabbitmq_manager

        self.message_receiver_thread = MessageReceiverThread(self.rabbitmq_manager.channel, 'prediction')
        self.message_receiver_thread.message_received.connect(self.update_message)
        self.message_receiver_thread.start()

        self.layout = QVBoxLayout(self)
  

        # Video Label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("background-color: black;min-height: 220px;")
        self.layout.addWidget(self.video_label)

        # Source Combo Box
        self.source_combo = QComboBox()
        self.source_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.layout.addWidget(self.source_combo)

        self.available_sources = self.get_available_sources()
        for source in self.available_sources:
            self.source_combo.addItem(source)

        self.source_combo.currentIndexChanged.connect(self.update_source)

        # Browse Button with Icon
        self.browse_button = QPushButton(icon('fa.folder-open', color='white', scale_factor=1.2), ' Browse Video')
        self.browse_button.clicked.connect(self.browse_video)
        self.layout.addWidget(self.browse_button)

        # Exercise Label
        self.exercise_label = QLabel('Exercise Prediction: None')
        self.exercise_label.setAlignment(Qt.AlignCenter)
        self.exercise_label.setFont(QFont('Consolas', 14, QFont.Bold))
        self.exercise_label.setStyleSheet("font-size: 20px; padding: 10px; font-weight: bold;")
        self.layout.addWidget(self.exercise_label)

        # Last Updated Label
        self.last_prediction_label = QLabel('Last Updated: None')
        self.last_prediction_label.setAlignment(Qt.AlignCenter)
        self.last_prediction_label.setFont(QFont('Consolas', 11))
        self.last_prediction_label.setStyleSheet("font-size: 12px;")
        self.layout.addWidget(self.last_prediction_label)

        # Control Panel Layout
        self.control_panel_layout = QHBoxLayout()
        self.enable_button = QPushButton(icon('fa.toggle-on', color='white'), ' Enable Pipeline')
        self.enable_button.clicked.connect(self.toggle_pipeline)
        self.control_panel_layout.addWidget(self.enable_button)
        self.layout.addLayout(self.control_panel_layout)

        # Set some margins for better spacing
        self.layout.setContentsMargins(20, 20, 20, 10)

        self.cap = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose.Pose()
        self.pose_processor = PoseProcessor()

        self.pipeline_enabled = False

        self.header_title = QLabel('GymGuard - v1.0')
        self.header_title.setStyleSheet("color: #333A73; font-size: 24px")
        self.header_title.setFont(QFont('Garamond', 14))
        self.header_title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.header_title)  

        self.setMinimumHeight(620)
        self.setMinimumWidth(600)

    def update_message(self, message):
        message = message.title().replace("_", " ")
        self.exercise_label.setText(f"Exercise Prediction: {message}")
        self.last_prediction_label.setText(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")

    def get_available_sources(self):
        available_sources = ["Camera " + str(i) for i in range(10)]
        available_sources.extend(["Video File"])
        return available_sources

    def update_source(self, new_index):
        if self.cap is not None:
            self.cap.release()

        if self.available_sources[new_index].startswith("Camera"):
            self.cap = cv2.VideoCapture(int(self.available_sources[new_index][-1]))
        else:
            self.cap = cv2.VideoCapture(self.video_file_path)

    def browse_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        self.video_file_path, _ = file_dialog.getOpenFileName(self, "Browse Video File", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options)
        if self.video_file_path:
            self.source_combo.setCurrentIndex(self.source_combo.findText("Video File"))

    def toggle_pipeline(self):
        self.pipeline_enabled = not self.pipeline_enabled
        if self.pipeline_enabled:
            self.enable_button.setText('Disable Pipeline')
        else:
            self.enable_button.setText('Enable Pipeline')

    def reset_pipeline(self):
        self.pipeline_enabled = False
        self.enable_button.setText('Enable Pipeline')

    def update_frame(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("End of video")
                break

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(rgb_image)

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )
            else:
                pass

            if self.pipeline_enabled:
                keypoints = self.pose_processor.process(results)
                if keypoints is not None:
                    self.make_request(keypoints)

            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap)
            self.video_label.setAlignment(Qt.AlignCenter)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def make_request(self, keypoints):
        payload = json.dumps({'features': keypoints})
        self.rabbitmq_manager.channel.basic_publish(exchange='', routing_key='hello', body=payload)
        print(" [x] Sent data to Pipeline:", len(keypoints))
        return True

    def closeEvent(self, event):
        self.thread.join()
        event.accept()


class VideoThread(QThread):
    frame_updated = pyqtSignal(QImage)

    def __init__(self, video_window):
        super(VideoThread, self).__init__()
        self.video_window = video_window

    def run(self):
        self.video_window.update_frame()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GymGuard - Pose Estimation and Exercise Prediction - v1.0")
        self.setWindowIcon(QIcon('logo.png'))


        rabbitmq_manager = RabbitMQManager()

        self.video_window = VideoWindow(rabbitmq_manager)
        self.setCentralWidget(self.video_window)

        self.video_window.source_combo.currentIndexChanged.connect(self.start_video_thread)

    def interrupt_video(self, new_index):
        if self.video_window.cap is not None:
            self.video_window.cap.release()
            self.video_window.video_label.clear()
            self.video_window.exercise_label.setText('Exercise Prediction: None')
            self.video_window.reset_pipeline()

    def start_video_thread(self, new_index):
        if self.video_window.cap is not None:
            self.video_thread = VideoThread(self.video_window)
            self.video_thread.frame_updated.connect(self.update_frame)
            self.video_thread.start()

    def update_frame(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.video_window.video_label.setPixmap(pixmap)
        self.video_window.video_label.setAlignment(Qt.AlignCenter)

    # make all things to destroy when close the window
    def closeEvent(self, event):
        self.video_window.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    with open('styles/glass.qss', 'r') as file:
        app.setStyleSheet(file.read())

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
