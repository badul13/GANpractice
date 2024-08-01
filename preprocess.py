import cv2
import os
from tqdm import tqdm

def preprocess_videos(video_a_path, video_b_path, output_dir, target_size=(1080, 1920)):
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

        with tqdm(total=total_frames, desc="Preprocessing videos") as pbar:
            while cap_a.isOpened() and cap_b.isOpened():
                ret_a, frame_a = cap_a.read()
                ret_b, frame_b = cap_b.read()

                if not ret_a or not ret_b:
                    break

                # 크기 조정
                frame_a = cv2.resize(frame_a, target_size)
                frame_b = cv2.resize(frame_b, target_size)

                cv2.imwrite(os.path.join(output_dir, f"A_{frame_id:04d}.png"), frame_a)
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
preprocess_videos('video_A.mp4', 'video_B.mp4', 'processed_frames', target_size=(1088,1920))