import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from pose_format import Pose

class TemporalInterpolatePose:
    def __init__(self, frames=64):
        self.frames = frames

    def __call__(self, data: torch.Tensor):
        T, V, C = data.shape
        if T == self.frames:
            return data

        data_reshaped = data.reshape(T, V * C).permute(1, 0).unsqueeze(0)
        data_interpolated = F.interpolate(
            data_reshaped,
            size=self.frames,
            mode='linear',
            align_corners=False,
        )
        return data_interpolated.squeeze(0).permute(1, 0).view(self.frames, V, C)

class RandomPoseNoise:
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, data: torch.Tensor):
        noise = torch.randn_like(data) * self.std
        return data + noise

class RandomPoseScale:
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale
    def __call__(self, data: torch.Tensor):
        scale = random.uniform(self.min_scale, self.max_scale)
        return data * scale

class RandomTemporalCrop:
    def __init__(self, frames=64):
        self.frames = frames

    def __call__(self, data: torch.Tensor):
        T, V, C = data.shape
        if T == self.frames:
            return data
        if T > self.frames:
            max_start = T - self.frames
            start_idx = random.randint(0, max_start)
            return data[start_idx: start_idx + self.frames, :, :]
        else:
            indices = torch.linspace(0, T - 1, steps=self.frames).long()
            return data[indices, :, :]

class PoseJoinSelect:
    def __init__(self):
        self.pose_landmarks = [
            "NOSE", "NECK", "RIGHT_EYE", "LEFT_EYE", "RIGHT_EAR", "LEFT_EAR",
            "RIGHT_SHOULDER", "LEFT_SHOULDER", "RIGHT_ELBOW", "LEFT_ELBOW",
            "RIGHT_WRIST", "LEFT_WRIST",
        ]
        self.hand_landmarks = [
            "WRIST", "INDEX_FINGER_TIP", "INDEX_FINGER_DIP", "INDEX_FINGER_PIP",
            "INDEX_FINGER_MCP", "MIDDLE_FINGER_TIP", "MIDDLE_FINGER_DIP",
            "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_MCP", "RING_FINGER_TIP",
            "RING_FINGER_DIP", "RING_FINGER_PIP", "RING_FINGER_MCP",
            "PINKY_DIP", "PINKY_TIP", "PINKY_PIP", "PINKY_MCP",
            "THUMB_TIP", "THUMB_IP", "THUMB_MCP", "THUMB_CMC"
        ]

    def __get_point(self, component: str, point: str, pose: Pose) -> np.ndarray:
        if point == "NECK":
            return np.zeros_like(pose.body.data[:, 0, 0, :2].data)
        idx = pose.header._get_point_index(component, point)
        return pose.body.data[:, 0, idx, :2].data

    def __call__(self, pose: Pose) -> torch.Tensor:
        data = []
        for point in self.pose_landmarks:
            data.append(self.__get_point("POSE_LANDMARKS", point, pose))
        for side in ["LEFT", "RIGHT"]:
            for point in self.hand_landmarks:
                data.append(self.__get_point(f"{side}_HAND_LANDMARKS", point, pose))
        # (num_landmarks, num_frames, 2) -> (num_frames, num_landmarks, 2)
        data = np.array(data).transpose((1, 0, 2))
        return torch.from_numpy(data)


class PoseNormalize:
    def __init__(self):
        self.BODY_LANDMARKS = [
            "nose", "neck", "rightEye", "leftEye", "rightEar", "leftEar",
            "rightShoulder", "leftShoulder", "rightElbow", "leftElbow",
            "rightWrist", "leftWrist"
        ]
        self.HAND_LANDMARKS = [
            "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
            "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
            "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ]

    def normalize_body(self, row: dict) -> dict:
        sequence_size = len(row["leftEar"])
        valid_sequence = True
        original_row = row
        last_starting_point, last_ending_point = None, None

        for sequence_index in range(sequence_size):
            if (row["leftShoulder"][sequence_index][0] == 0 or row["rightShoulder"][sequence_index][0] == 0) and (
                    row["neck"][sequence_index][0] == 0 or row["nose"][sequence_index][0] == 0):
                if not last_starting_point:
                    valid_sequence = False
                    continue
                else:
                    starting_point, ending_point = last_starting_point, last_ending_point
            else:
                if (row["leftShoulder"][sequence_index][0] != 0 and row["rightShoulder"][sequence_index][0] != 0):
                    left_shoulder = (row["leftShoulder"][sequence_index][0], row["leftShoulder"][sequence_index][1])
                    right_shoulder = (row["rightShoulder"][sequence_index][0], row["rightShoulder"][sequence_index][1])
                    shoulder_distance = (((left_shoulder[0] - right_shoulder[0]) ** 2) + (
                                (left_shoulder[1] - right_shoulder[1]) ** 2)) ** 0.5
                    head_metric = shoulder_distance
                else:
                    neck = (row["neck"][sequence_index][0], row["neck"][sequence_index][1])
                    nose = (row["nose"][sequence_index][0], row["nose"][sequence_index][1])
                    neck_nose_distance = (((neck[0] - nose[0]) ** 2) + ((neck[1] - nose[1]) ** 2)) ** 0.5
                    head_metric = neck_nose_distance

                starting_point = [
                    row["neck"][sequence_index][0] - 3 * head_metric,
                    row["leftEye"][sequence_index][1] + head_metric,
                ]
                ending_point = [
                    row["neck"][sequence_index][0] + 3 * head_metric,
                    starting_point[1] - 6 * head_metric,
                ]
                last_starting_point, last_ending_point = starting_point, ending_point

            if starting_point[0] < 0: starting_point[0] = 0
            if starting_point[1] < 0: starting_point[1] = 0
            if ending_point[0] < 0: ending_point[0] = 0
            if ending_point[1] < 0: ending_point[1] = 0

            for identifier in self.BODY_LANDMARKS:  # Dùng self.BODY_LANDMARKS
                key = identifier
                if row[key][sequence_index][0] == 0: continue
                if any([(ending_point[0] - starting_point[0]) == 0, (starting_point[1] - ending_point[1]) == 0]):
                    logging.info(f"Problematic normalization with {key}")
                    valid_sequence = False
                    break

                normalized_x = (row[key][sequence_index][0] - starting_point[0]) / (ending_point[0] - starting_point[0])
                normalized_y = (row[key][sequence_index][1] - ending_point[1]) / (starting_point[1] - ending_point[1])

                row[key][sequence_index] = list(row[key][sequence_index])
                row[key][sequence_index][0] = normalized_x
                row[key][sequence_index][1] = normalized_y

        if valid_sequence: return row
        return original_row

    def normalize_hand(self, row: dict) -> dict:
        hand_landmarks = {0: [], 1: []}
        range_hand_size = 1
        if "wrist_1" in row.keys():
            range_hand_size = 2

        for identifier in self.HAND_LANDMARKS:
            for hand_index in range(range_hand_size):
                hand_landmarks[hand_index].append(identifier + "_" + str(hand_index))

        for hand_index in range(range_hand_size):
            sequence_size = len(row["wrist_" + str(hand_index)])
            for sequence_index in range(sequence_size):
                landmarks_x_values = [row[key][sequence_index][0] for key in hand_landmarks[hand_index] if
                                      row[key][sequence_index][0] != 0]
                landmarks_y_values = [row[key][sequence_index][1] for key in hand_landmarks[hand_index] if
                                      row[key][sequence_index][1] != 0]

                if not landmarks_x_values or not landmarks_y_values: continue

                width = max(landmarks_x_values) - min(landmarks_x_values)
                height = max(landmarks_y_values) - min(landmarks_y_values)

                if width > height:
                    delta_x = 0.1 * width
                    delta_y = delta_x + ((width - height) / 2)
                else:
                    delta_y = 0.1 * height
                    delta_x = delta_y + ((height - width) / 2)

                starting_point = (min(landmarks_x_values) - delta_x, min(landmarks_y_values) - delta_y)
                ending_point = (max(landmarks_x_values) + delta_x, max(landmarks_y_values) + delta_y)

                for identifier in self.HAND_LANDMARKS:
                    key = identifier + "_" + str(hand_index)
                    if (row[key][sequence_index][0] == 0 or (ending_point[0] - starting_point[0]) == 0 or (
                            starting_point[1] - ending_point[1]) == 0):
                        continue

                    normalized_x = (row[key][sequence_index][0] - starting_point[0]) / (
                                ending_point[0] - starting_point[0])
                    normalized_y = (row[key][sequence_index][1] - starting_point[1]) / (
                                ending_point[1] - starting_point[1])

                    row[key][sequence_index] = list(row[key][sequence_index])
                    row[key][sequence_index][0] = normalized_x
                    row[key][sequence_index][1] = normalized_y

        return row

    def _tensor_to_dict(self, data_tensor: torch.Tensor) -> dict:

        T = data_tensor.shape[0]
        row_dict = {}

        for i, name in enumerate(self.BODY_LANDMARKS):
            row_dict[name] = data_tensor[:, i, :].tolist()

        for i, name in enumerate(self.HAND_LANDMARKS):
            row_dict[name + "_0"] = data_tensor[:, 12 + i, :].tolist()

        for i, name in enumerate(self.HAND_LANDMARKS):
            row_dict[name + "_1"] = data_tensor[:, 33 + i, :].tolist()

        return row_dict

    def _dict_to_tensor(self, row_dict: dict, original_tensor: torch.Tensor) -> torch.Tensor:
        out_tensor = original_tensor.clone()

        for i, name in enumerate(self.BODY_LANDMARKS):
            out_tensor[:, i, :] = torch.tensor(row_dict[name])

        for i, name in enumerate(self.HAND_LANDMARKS):
            out_tensor[:, 12 + i, :] = torch.tensor(row_dict[name + "_0"])

        for i, name in enumerate(self.HAND_LANDMARKS):
            out_tensor[:, 33 + i, :] = torch.tensor(row_dict[name + "_1"])

        return out_tensor - 0.5

    def __call__(self, data_tensor: torch.Tensor):
        row_dict = self._tensor_to_dict(data_tensor)
        row_dict = self.normalize_body(row_dict)
        row_dict = self.normalize_hand(row_dict)
        return self._dict_to_tensor(row_dict, data_tensor)