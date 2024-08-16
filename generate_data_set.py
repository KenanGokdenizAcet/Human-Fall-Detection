import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import glob
import os

def generate_csv_from_images(images_path: str, input_csv_path: str, output_path: str):
    source_csv = pd.read_csv(input_csv_path)
    tag_values = source_csv['Tag'].unique()
    idle = tag_values[1]
    falling = tag_values[2]
    fell = tag_values[3]

    sorted_images_path = sorted(glob.glob(os.path.join(images_path, '*.png')))

    data_frame = pd.DataFrame()

    # initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    frame_count = 0

    frame_queue = []

    main_column_names = [
        'head_x',
        'head_y',
        'l_shoulder_x',
        'l_shoulder_y',
        'l_belly_x',
        'l_belly_y',
        'l_knee_x',
        'l_knee_y',
        'r_shoulder_x',
        'r_shoulder_y',
        'r_belly_x',
        'r_belly_y',
        'r_knee_x',
        'r_knee_y']

    for image_path in sorted_images_path:
        frame = cv2.imread(image_path)

        results = pose.process(frame)

        df = []

        if results.pose_landmarks is not None:
            # Extract key points
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of head, shoulder, belly, and knees
            head_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
            head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y

            l_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
            l_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            l_belly_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
            l_belly_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            l_knee_x = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x
            l_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y

            r_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
            r_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            r_belly_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
            r_belly_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            r_knee_x = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
            r_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

            df.append({'frame': frame_count,
                       'head_x': head_x,
                       'head_y': head_y,
                       'l_shoulder_x': l_shoulder_x,
                       'l_shoulder_y': l_shoulder_y,
                       'l_belly_x': l_belly_x,
                       'l_belly_y': l_belly_y,
                       'l_knee_x': l_knee_x,
                       'l_knee_y': l_knee_y,
                       'r_shoulder_x': r_shoulder_x,
                       'r_shoulder_y': r_shoulder_y,
                       'r_belly_x': r_belly_x,
                       'r_belly_y': r_belly_y,
                       'r_knee_x': r_knee_x,
                       'r_knee_y': r_knee_y})

            df = pd.DataFrame(df)

            # adding new columns for difference of previous frames
            for i in range(4):
                for col in main_column_names:
                    new_column_name = f'{col}diff_{i + 1}frame_before'
                    df[new_column_name] = np.nan

            try:
                if source_csv.iloc[frame_count + 1, -1] == idle:
                    df["label"] = 0
                elif source_csv.iloc[frame_count + 1, -1] == falling:
                    df["label"] = 1
                else:
                    df["label"] = -1
            except IndexError:
                break

            if len(frame_queue) <= 4:
                for i in range(min(len(frame_queue), 4)):
                    for col in main_column_names:
                        df[f"{col}diff_{i + 1}frame_before"] = df[col] - frame_queue[-(i + 1)][col]
                frame_queue.append(df)
            else:
                frame_queue.pop(0)
                for i in range(4):
                    for col in main_column_names:
                        df[f"{col}diff_{i + 1}frame_before"] = df[col] - frame_queue[-(i + 1)][col]
                frame_queue.append(df)

            data_frame = pd.concat([data_frame, df], ignore_index=True, sort=False)

        frame_count += 1

    data_frame.to_csv(output_path, index=False)



def generate_all(path: str, output_path: str):
    dirs = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            dirs.append(item)
    dirs.sort()
    for d in dirs:
        for i in range(3):
            for j in range(2):
                images_path = path + "/" + d + "/" + d + "Trial" + str(i+1) + "Camera" + str(j+1)
                csv_src_path = path + "/" + d + "/" + d + "Trial" + str(i+1) + ".csv"
                csv_dst_path = output_path + "/" + d + "Trial" + str(i+1) + "Camera" + str(j+1) + "labelled.csv"
                generate_csv_from_images(images_path, csv_src_path, csv_dst_path)
