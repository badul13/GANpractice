import cv2
import os
from tqdm import tqdm
from mtcnn import MTCNN
import numpy as np
import dlib


def align_face(img, landmarks):
    # 기준 점: 두 눈의 중심과 코의 중심
    left_eye_center = np.mean(landmarks[36:42], axis=0)
    right_eye_center = np.mean(landmarks[42:48], axis=0)
    nose_center = landmarks[30]

    eyes_center = (left_eye_center + right_eye_center) / 2.0

    # 눈의 각도 계산
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(dy, dx)) - 180

    # 회전 및 변환 행렬 계산
    eyes_distance = np.sqrt(dx ** 2 + dy ** 2)
    desired_eyes_distance = 70
    scale = desired_eyes_distance / eyes_distance

    center = (int(eyes_center[0]), int(eyes_center[1]))
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 얼굴 정렬
    output_size = (256, 256)
    aligned_face = cv2.warpAffine(img, M, output_size, flags=cv2.INTER_CUBIC)

    return aligned_face


def preprocess_videos(video_a_path, video_b_path, output_dir, target_size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        cap_a = cv2.VideoCapture(video_a_path)
        cap_b = cv2.VideoCapture(video_b_path)

        if not cap_a.isOpened():
            raise IOError(f"Cannot open video file: {video_a_path}")
        if not cap_b.isOpened():
            raise IOError(f"Cannot open video file: {video_b_path}")

        frame_id = 0
        total_frames = min(int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT)))

        detector = MTCNN()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # dlib의 68 landmarks 모델 필요

        with tqdm(total=total_frames, desc="Preprocessing videos") as pbar:
            while cap_a.isOpened() and cap_b.isOpened():
                ret_a, frame_a = cap_a.read()
                ret_b, frame_b = cap_b.read()

                if not ret_a or not ret_b:
                    break

                # 얼굴 검출 및 정렬
                faces_a = detector.detect_faces(frame_a)
                faces_b = detector.detect_faces(frame_b)

                if len(faces_a) == 0 or len(faces_b) == 0:
                    continue

                # 첫 번째 얼굴만 사용
                face_a = faces_a[0]
                face_b = faces_b[0]

                x, y, width, height = face_a['box']
                face_a_img = frame_a[y:y + height, x:x + width]
                face_a_img_rgb = cv2.cvtColor(face_a_img, cv2.COLOR_BGR2RGB)
                rect_a = dlib.rectangle(0, 0, face_a_img.shape[1], face_a_img.shape[0])
                landmarks_a = predictor(face_a_img_rgb, rect_a)
                landmarks_a = np.array([[p.x, p.y] for p in landmarks_a.parts()])

                x, y, width, height = face_b['box']
                face_b_img = frame_b[y:y + height, x:x + width]
                face_b_img_rgb = cv2.cvtColor(face_b_img, cv2.COLOR_BGR2RGB)
                rect_b = dlib.rectangle(0, 0, face_b_img.shape[1], face_b_img.shape[0])
                landmarks_b = predictor(face_b_img_rgb, rect_b)
                landmarks_b = np.array([[p.x, p.y] for p in landmarks_b.parts()])

                aligned_face_a = align_face(face_a_img, landmarks_a)
                aligned_face_b = align_face(face_b_img, landmarks_b)

                # 합성
                mask = np.zeros(aligned_face_b.shape, dtype=np.uint8)
                mask = cv2.fillConvexPoly(mask, cv2.convexHull(landmarks_b), (255, 255, 255))

                center = (aligned_face_b.shape[1] // 2, aligned_face_b.shape[0] // 2)
                blended_face = cv2.seamlessClone(aligned_face_a, aligned_face_b, mask, center, cv2.NORMAL_CLONE)

                frame_b[y:y + height, x:x + width] = cv2.resize(blended_face, (width, height))

                # 저장
                cv2.imwrite(os.path.join(output_dir, f"B_{frame_id:04d}.png"), frame_b)
                frame_id += 1

                pbar.update(1)

        print(f"Processed {frame_id} frames.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        if 'cap_a' in locals():
            cap_a.release()
        if 'cap_b' in locals():
            cap_b.release()


# 예시 사용법 (비디오 파일 이름을 실제 파일 이름으로 변경하세요)
preprocess_videos('video_A.mp4', 'video_B.mp4', 'processed_frames', target_size=(256, 256))
