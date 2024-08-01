import os
import cv2
import dlib
import numpy as np
import logging
from imutils import face_utils
import tensorflow as tf

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Dlib의 얼굴 검출기와 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None

    rect = rects[0]
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    (x, y, w, h) = face_utils.rect_to_bb(rect)
    face = frame[y:y + h, x:x + w]

    face = cv2.resize(face, (128, 128))
    face = face / 255.0  # Normalize to [0, 1]

    return face

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        face = extract_face(frame)
        if face is not None:
            frames.append(face)
    cap.release()
    return np.array(frames)

def load_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        logging.warning(f"Failed to read image {image_path}")
        return None
    face = extract_face(frame)
    return face if face is not None else None

def load_data_from_folder(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        try:
            if filename.endswith(('.mp4', '.avi', '.mov')):
                data.extend(load_video(path))
            elif filename.endswith(('.jpg', '.jpeg', '.png')):
                face = load_image(path)
                if face is not None:
                    data.append(face)
        except Exception as e:
            logging.warning(f"Failed to process {path}: {e}")
    return np.array(data)

def augment_images(images):
    augmented_images = []
    for img in images:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
        img = tf.image.random_hue(img, max_delta=0.05)
        img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
        augmented_images.append(img)
    return np.array(augmented_images)

folder_path = 'data_folder'  # 사용할 폴더 경로
data_frames = load_data_from_folder(folder_path)
logging.info(f"Loaded {len(data_frames)} frames.")

augmented_data_frames = augment_images(data_frames)
logging.info(f"Augmented to {len(augmented_data_frames)} frames.")

# 데이터 저장
np.save('preprocessed_data.npy', augmented_data_frames)
logging.info(f"Preprocessed data saved to 'preprocessed_data.npy'.")
