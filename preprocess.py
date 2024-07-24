import cv2
import os
from mtcnn import MTCNN


def preprocess_videos(video_path_A, video_path_B, output_dir, frame_size=(128, 128)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detector = MTCNN()

    def extract_faces(video_path, label):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0

        while frame_index < frame_count:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector.detect_faces(frame)
            if results:
                face = results[0]['box']
                x, y, w, h = face
                face_img = frame[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, frame_size)
                cv2.imwrite(os.path.join(output_dir, f"{label}_{frame_index:04d}.png"), face_img)

            frame_index += 1

        cap.release()

    extract_faces(video_path_A, 'A')
    extract_faces(video_path_B, 'B')


# 예시 사용법
preprocess_videos('video_A.mp4', 'video_B.mp4', 'processed_frames')
