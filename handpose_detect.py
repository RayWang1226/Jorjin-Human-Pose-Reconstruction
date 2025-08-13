import os
import cv2
import json
import numpy as np
from tqdm import tqdm

import torch
import romp
import mediapipe as mp

mp_hands = mp.solutions.hands


def calculate_finger_angles(landmarks):
    """Calculate angles between finger segments to determine pose features"""
    angles = {}

    def calculate_angle(p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0:
            return 0
        dot_product = np.dot(v1, v2)
        angle = np.arccos(
            np.clip(dot_product / (v1_norm * v2_norm), -1.0, 1.0))
        return np.degrees(angle)

    lm = landmarks.landmark
    angles['thumb_fold'] = calculate_angle(lm[2], lm[3], lm[4])
    angles['index_fold'] = calculate_angle(lm[5], lm[6], lm[8])
    angles['middle_fold'] = calculate_angle(lm[9], lm[10], lm[12])
    angles['ring_fold'] = calculate_angle(lm[13], lm[14], lm[16])
    angles['pinky_fold'] = calculate_angle(lm[17], lm[18], lm[20])

    thumb_tip = np.array([lm[4].x, lm[4].y, lm[4].z])
    index_tip = np.array([lm[8].x, lm[8].y, lm[8].z])
    angles['thumb_index_dist'] = np.linalg.norm(thumb_tip - index_tip)

    return angles


def is_fist(hand_landmarks, handedness):
    angles = calculate_finger_angles(hand_landmarks)
    return all(angles[f'{finger}_fold'] < 120 for finger in ['thumb', 'index', 'middle', 'ring', 'pinky'])


def is_thumbs_up(hand_landmarks, handedness):
    lm = hand_landmarks.landmark
    angles = calculate_finger_angles(hand_landmarks)
    thumb_up = lm[4].y < lm[3].y if handedness == "Right" else lm[4].y > lm[3].y
    fingers_folded = all(angles[f'{finger}_fold'] < 120 for finger in [
                         'index', 'middle', 'ring', 'pinky'])
    return thumb_up and fingers_folded


def is_pointing(hand_landmarks, handedness):
    angles = calculate_finger_angles(hand_landmarks)
    index_extended = angles['index_fold'] > 160
    other_fingers_folded = all(angles[f'{finger}_fold'] < 120 for finger in [
                               'middle', 'ring', 'pinky'])
    return index_extended and other_fingers_folded


def is_yeah_pose(hand_landmarks, handedness):
    angles = calculate_finger_angles(hand_landmarks)
    index_extended = angles['index_fold'] > 150
    middle_extended = angles['middle_fold'] > 150
    other_fingers_folded = all(
        angles[f'{finger}_fold'] < 120 for finger in ['ring', 'pinky'])
    return index_extended and middle_extended and other_fingers_folded


def is_ok_pose(hand_landmarks, handedness):
    angles = calculate_finger_angles(hand_landmarks)
    thumb_index_close = angles['thumb_index_dist'] < 0.1
    other_fingers_extended = all(angles[f'{finger}_fold'] > 140 for finger in [
                                 'middle', 'ring', 'pinky'])
    return thumb_index_close and other_fingers_extended


def detect_hand_pose(hand_landmarks, handedness):
    if is_thumbs_up(hand_landmarks, handedness):
        return "thumbs_up"
    elif is_pointing(hand_landmarks, handedness):
        return "pointing"
    elif is_yeah_pose(hand_landmarks, handedness):
        return "yeah"
    elif is_ok_pose(hand_landmarks, handedness):
        return "ok"
    elif is_fist(hand_landmarks, handedness):
        return "fist"
    else:
        return "unknown"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--consecutive_frames", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Processing {total_frames} frames from video...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(args.output_dir, "output_pose_overlay.mp4")
    out_video = cv2.VideoWriter(
        out_path, fourcc, fps, (frame_width, frame_height))

    hands = mp_hands.Hands(model_complexity=1, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize tracking dictionaries for both hands
    hand_pose_tracking = {
        "Right": {
            "current_pose": "unknown",
            "confirmed_pose": "unknown",
            "consecutive_count": 0,
            "frames_since_detection": 0,  # New field to track absence
            "persistent_pose": "unknown"   # New field for the persistent pose
        },
        "Left": {
            "current_pose": "unknown",
            "confirmed_pose": "unknown",
            "consecutive_count": 0,
            "frames_since_detection": 0,  # New field to track absence
            "persistent_pose": "unknown"   # New field for the persistent pose
        }
    }

    frame_idx = 0

    # Maximum frames to keep a pose when hand is not visible
    MAX_FRAMES_KEEP_POSE = 60  # About 2 seconds at 30fps

    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_result = hands.process(rgb_frame)

            hands_detected = {"Right": False, "Left": False}

            if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                for landmarks, handedness_info in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
                    # Get hand type ('Right' or 'Left')
                    handedness = handedness_info.classification[0].label
                    hands_detected[handedness] = True

                    # Reset frames since detection counter since we found this hand
                    hand_pose_tracking[handedness]["frames_since_detection"] = 0

                    # Detect which pose this hand is making
                    detected_pose = detect_hand_pose(landmarks, handedness)

                    # Update tracking for this hand
                    if detected_pose == hand_pose_tracking[handedness]["current_pose"]:
                        # Increment consecutive counter
                        hand_pose_tracking[handedness]["consecutive_count"] += 1
                    else:
                        # Reset counter for new pose
                        hand_pose_tracking[handedness]["current_pose"] = detected_pose
                        hand_pose_tracking[handedness]["consecutive_count"] = 1

                    # Check if we've reached the threshold for confirmed pose change
                    if (hand_pose_tracking[handedness]["consecutive_count"] >= args.consecutive_frames and
                            hand_pose_tracking[handedness]["confirmed_pose"] != detected_pose):
                        # Update the confirmed pose
                        hand_pose_tracking[handedness]["confirmed_pose"] = detected_pose
                        # Also update the persistent pose
                        hand_pose_tracking[handedness]["persistent_pose"] = detected_pose

                        # Report the new confirmed pose
                        if handedness == "Right":
                            print(f"Left {detected_pose}.")
                        else:
                            print(f"Right {detected_pose}.")

                    # Overlay confirmed pose in top-left corner of display frame
                    # Switch left and right due to mediapipe settings
                    if handedness == "Right":
                        overlay_text = f"Left: {hand_pose_tracking[handedness]['persistent_pose']}"
                    else:
                        overlay_text = f"Right: {hand_pose_tracking[handedness]['persistent_pose']}"
                    cv2.putText(display_frame, overlay_text,
                                (10, 30 if handedness == "Left" else 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            # Handle hands not detected in this frame
            for hand in ["Right", "Left"]:
                if not hands_detected[hand]:
                    hand_pose_tracking[hand]["consecutive_count"] = 0
                    hand_pose_tracking[hand]["current_pose"] = "unknown"
                    # Increment the counter for frames since last detection
                    hand_pose_tracking[hand]["frames_since_detection"] += 1

                    # Only reset persistent pose after long absence
                    if hand_pose_tracking[hand]["frames_since_detection"] > MAX_FRAMES_KEEP_POSE:
                        hand_pose_tracking[hand]["persistent_pose"] = "unknown"

            out_video.write(display_frame)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    hands.close()
    out_video.release()

    print(f"[DONE] Processed all frames. Saved output video to {out_path}")

