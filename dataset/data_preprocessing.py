import torch
import numpy as np
from pose_format import Pose

class PointPoseSelect:
    def __init__(self):
        self.pose_landmarks = ["NOSE", "NECK", "RIGHT_EYE", "LEFT_EYE", "RIGHT_EAR", "LEFT_EAR",
                               "RIGHT_SHOULDER", "LEFT_SHOULDER", "RIGHT_ELBOW", "LEFT_ELBOW",
                               "RIGHT_WRIST", "LEFT_WRIST"]
        self.hand_landmarks = ["WRIST", "INDEX_FINGER_TIP", "INDEX_FINGER_DIP", "INDEX_FINGER_PIP",
                               "INDEX_FINGER_MCP", "MIDDLE_FINGER_TIP", "MIDDLE_FINGER_DIP",
                               "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_MCP", "RING_FINGER_TIP",
                               "RING_FINGER_DIP", "RING_FINGER_PIP", "RING_FINGER_MCP",
                               "PINKY_DIP", "PINKY_TIP", "PINKY_PIP", "PINKY_MCP",
                               "THUMB_TIP", "THUMB_IP", "THUMB_MCP", "THUMB_CMC"]

    def __get_point(self, components: str, point: str, pose: Pose) -> np.ndarray:
        if point == 'NECK':
            return np.zeros_like(pose.body.data[:, 0, 0, :2].data)
        idx = pose.header._get_point_index(components, point)
        return pose.body.data[:, 0, idx, :2].data

    def __call__(self, pose: Pose) -> torch.Tensor:
        data = []
        for point in self.pose_landmarks:
            data.append(self.__get_point(components="POSE_LANDMARKS", point=point, pose=pose))
        for side in ['LEFT', 'RIGHT']:
            for point in self.hand_landmarks:
                data.append(self.__get_point(components=f'{side}_HAND_LANDMARKS', point=point, pose=pose))

        # (num_landmarks, num_frames, 2) -> (num_frames, num_landmarks, 2)
        data = np.array(data).transpose(1, 0, 2)
        return torch.from_numpy(data)


class BodyPoseNormalize:
    def __init__(self):
        self.pose_landmarks = ["nose", "neck", "rightEye", "leftEye", "rightEar", "leftEar",
                               "rightShoulder", "leftShoulder", "rightElbow", "leftElbow",
                               "rightWrist", "leftWrist"]

    def __call__(self, row: dict) -> dict:
        sequence_size = len(row['leftEar'])
        valid_sequence = True
        original_row = row
        last_starting_point, last_ending_point = None, None

        for sequence_index in range(sequence_size):
            if (row['leftShoulder'][sequence_index][0] == 0 or row['rightShoulder'][sequence_index][0] == 0) and \
                    (row['neck'][sequence_index][0] == 0 or row['nose'][sequence_index][0] == 0):
                if not last_starting_point:
                    valid_sequence = False
                    continue
                else:
                    starting_point, ending_point = last_starting_point, last_ending_point
            else:
                if row["leftShoulder"][sequence_index][0] != 0 and row["rightShoulder"][sequence_index][0] != 0:
                    ls, rs = row["leftShoulder"][sequence_index], row["rightShoulder"][sequence_index]
                    head_metric = ((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2) ** 0.5
                else:
                    neck, nose = row["neck"][sequence_index], row["nose"][sequence_index]
                    head_metric = ((neck[0] - nose[0]) ** 2 + (neck[1] - nose[1]) ** 2) ** 0.5

                starting_point = [row["neck"][sequence_index][0] - 3 * head_metric,
                                  row["leftEye"][sequence_index][1] + head_metric]
                ending_point = [row["neck"][sequence_index][0] + 3 * head_metric, starting_point[1] - 6 * head_metric]
                last_starting_point, last_ending_point = starting_point, ending_point

            for identifier in self.pose_landmarks:
                if row[identifier][sequence_index][0] == 0: continue

                denom_x = ending_point[0] - starting_point[0]
                denom_y = starting_point[1] - ending_point[1]
                if denom_x == 0 or denom_y == 0: continue

                # (x - min) / (max - min)
                normalized_x = (row[identifier][sequence_index][0] - starting_point[0]) / denom_x
                normalized_y = (row[identifier][sequence_index][1] - ending_point[1]) / denom_y
                row[identifier][sequence_index] = [normalized_x, normalized_y]

        return row if valid_sequence else original_row

class UniformPosePad:
    def __init__(self, num_frames: int = 150):
        self.num_frames = num_frames

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # data shape: (Frames, Landmarks, Channels)
        L = data.shape[0]
        padded_data = torch.zeros((self.num_frames, data.shape[1], data.shape[2]))

        if L < self.num_frames:
            padded_data[:L, :, :] = data
            rest = self.num_frames - L
            num = int(np.ceil(rest / L))
            pad = torch.cat([data for _ in range(num)], dim=0)[:rest]
            padded_data[L:, :, :] = pad
        else:
            padded_data = data[:self.num_frames, :, :]

        return padded_data

class PoseMotionStream:
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # Motion = Frame(t+1) - Frame(t)
        T = data.shape[0]
        motion_data = torch.zeros_like(data)
        motion_data[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
        return motion_data

class HandPoseNormalize:
    def __init__(self):
        self.finger_keys = ["wrist", "thumb", "index", "middle", "ring", "pinky"]

    def __call__(self, row: dict) -> dict:
        hand_count = 2 if "wrist_1" in row.keys() else 1

        for h_idx in range(hand_count):
            x_vals = [row[k][i][0] for k in row.keys() if f"_{h_idx}" in k for i in range(len(row[k])) if
                      row[k][i][0] != 0]
            y_vals = [row[k][i][1] for k in row.keys() if f"_{h_idx}" in k for i in range(len(row[k])) if
                      row[k][i][1] != 0]

            if not x_vals or not y_vals: continue

            width, height = max(x_vals) - min(x_vals), max(y_vals) - min(y_vals)
            delta = 0.1 * max(width, height)

            start_p = (min(x_vals) - delta, min(y_vals) - delta)
            end_p = (max(x_vals) + delta, max(y_vals) + delta)

            for key in row.keys():
                if f"_{h_idx}" in key:
                    for i in range(len(row[key])):
                        if row[key][i][0] == 0: continue
                        # (value - min) / (max - min)
                        norm_x = (row[key][i][0] - start_p[0]) / (end_p[0] - start_p[0])
                        norm_y = (row[key][i][1] - start_p[1]) / (end_p[1] - start_p[1])
                        row[key][i] = [norm_x, norm_y]
        return row

class PoseBoneStream:
    def __init__(self):
        # (Joint2_idx, Joint1_idx)
        self.bone_pairs = [
            (5, 6), (5, 7), (6, 8), (8, 10), (7, 9), (9, 11),
            (12, 13), (12, 14), (12, 16), (12, 18), (12, 20),
            (14, 15), (16, 17), (18, 19), (20, 21),
            (22, 23), (22, 24), (22, 26), (22, 28), (22, 30),
            (24, 25), (26, 27), (28, 29), (30, 31),
            (10, 12), (11, 22)
        ]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # Bone = Joint2 - Joint1
        bone_data = data.clone()
        for v1, v2 in self.bone_pairs:
            if v2 < data.shape[1]:
                bone_data[:, v2, :] = data[:, v2, :] - data[:, v1, :]
        return bone_data

class PoseTensorToDict:
    def __init__(self, landmark_names: list):
        self.names = landmark_names

    def __call__(self, data: torch.Tensor) -> dict:
        # data shape: (Frames, Landmarks, Channels)
        data_np = data.numpy()
        output = {}
        for i, name in enumerate(self.names):
            if i < data_np.shape[1]:
                output[name] = data_np[:, i].tolist()
        return output

class GlobalPoseNormalize:
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # data - mean(data)
        mean_pos = data.mean(dim=(0, 1), keepdim=True)
        return data - mean_pos

class PoseShift:
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # data - 0.5
        return data - 0.5