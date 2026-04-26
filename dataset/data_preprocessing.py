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

class UniformTemporalInterpolatePose:
    def __init__(self, frames=64):
        self.frames = frames

    def __call__(self, data: torch.Tensor):
        T, V, C = data.shape
        if T == self.frames:
            return data
        indices = torch.linspace(0, T - 1, steps=self.frames).long()
        return data[indices, :, :]

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

class GlobalPoseNormalize:
    def __call__(self, data: torch.Tensor):
        mean_pos = data.mean(dim=(0, 1), keepdim=True)
        data = data - mean_pos

        std_pos = data.std(dim=(0, 1), keepdim=True)
        data = data / (std_pos + 1e-6)

        return data


class PointPoseSelect:
    def __init__(self):
        self.hand_landmarks = [
            "WRIST", "INDEX_FINGER_TIP", "INDEX_FINGER_DIP", "INDEX_FINGER_PIP",
            "INDEX_FINGER_MCP", "MIDDLE_FINGER_TIP", "MIDDLE_FINGER_DIP",
            "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_MCP", "RING_FINGER_TIP",
            "RING_FINGER_DIP", "RING_FINGER_PIP", "RING_FINGER_MCP",
            "PINKY_DIP", "PINKY_TIP", "PINKY_PIP", "PINKY_MCP",
            "THUMB_TIP", "THUMB_IP", "THUMB_MCP", "THUMB_CMC"
        ]

    def __call__(self, pose: Pose) -> torch.Tensor:
        data = []
        for side in ['LEFT', 'RIGHT']:
            for point in self.hand_landmarks:
                try:
                    idx = pose.header._get_point_index(f'{side}_HAND_LANDMARKS', point)
                    data.append(pose.body.data[:, 0, idx, :].data)
                except:
                    data.append(np.zeros((pose.body.data.shape[0], 3)))

        # (42, T, 3) -> (T, 42, 3)
        data = np.array(data).transpose(1, 0, 2)
        return torch.from_numpy(data).float()


from torchvision.transforms import Compose
train_transforms = Compose([
    RandomTemporalCrop(frames=64),
    GlobalPoseNormalize()
])

val_test_transforms = Compose([
    TemporalInterpolatePose(frames=64),
    GlobalPoseNormalize()
])