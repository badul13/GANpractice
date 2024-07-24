import cv2
import os
from tqdm import tqdm

def preprocess_videos(video_a_path, video_b_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap_a = cv2.VideoCapture(video_a_path)
    cap_b = cv2.VideoCapture(video_b_path)

    frame_id = 0
    total_frames = min(int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_width = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tqdm(total=total_frames, desc="Preprocessing videos") as pbar:
        while cap_a.isOpened() and cap_b.isOpened():
            ret_a, frame_a = cap_a.read()
            ret_b, frame_b = cap_b.read()

            if not ret_a or not ret_b:
                break

            cv2.imwrite(os.path.join(output_dir, f"A_{frame_id:04d}.png"), frame_a)
            cv2.imwrite(os.path.join(output_dir, f"B_{frame_id:04d}.png"), frame_b)
            frame_id += 1

            pbar.update(1)

    cap_a.release()
    cap_b.release()

# 예시 사용법
preprocess_videos('video_a.mp4', 'video_b.mp4', 'processed_frames')
