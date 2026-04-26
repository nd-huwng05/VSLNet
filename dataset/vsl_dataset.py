import os
from json.decoder import NaN

import pandas as pd
import torch
from pose_format import Pose
from torch.utils.data import Dataset

from dataset.data_preprocessing import PointPoseSelect, train_transforms, val_test_transforms


class VSLPoseDataset(Dataset):
    def __init__(self, root_dir, split: str = 'train', views=["front_view", "left_view", "right_view"], transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.point_selector = PointPoseSelect()

        gloss_df = pd.read_csv(os.path.join(self.root_dir, f"gloss.csv"))
        self.gloss2id = dict(zip(gloss_df['gloss'], gloss_df['id']))
        self.data_df = self._build_and_split(views)

    def _build_and_split(self, views):
        samples = []
        for view in views:
            json_view = os.path.join(self.root_dir, f"{view}.json")
            if not os.path.exists(json_view): continue
            df = pd.read_json(
                json_view,
                encoding='utf-8',
                dtype={
                    "video_id": "string",
                    "signer_id": "string",
                    "fps": "int",
                    "resolution": "int",
                    "length": "float",
                    "gloss": "string",
                }
            )
            df["view"] = view
            df["gloss_id"] = df["gloss"].map(self.gloss2id)
            df["pose_path"] = df['video_id'].apply(
                lambda x: os.path.join(self.root_dir, f"{view}", f"{x}.pose")
            )
            samples.append(df)

        if not samples:
            raise FileNotFoundError("Not found view json file")

        df = pd.concat(samples, ignore_index=True)

        common_signer_ids = {
            "020": (views[0], views[1], views[2]),
            "014": (views[1], views[2], views[0]),
            "015": (views[2], views[0], views[1]),
        }
        val_unique_signer_ids = ["007"]
        test_unique_signer_ids = ["024"]
        val_test_common_signer_ids = ["009"]
        train_not_unique_signer_ids = (
            val_unique_signer_ids + test_unique_signer_ids + val_test_common_signer_ids + list(common_signer_ids.keys())
        )
        view_ids = list(df["view"].unique())
        val_test_df = df[df["signer_id"].isin(val_test_common_signer_ids)]
        val_df = val_test_df.groupby(["gloss_id", "view"], group_keys=False).apply(
            lambda x: x.sample(frac=0.5, random_state=42))
        test_df = val_test_df[~val_test_df.index.isin(val_df.index)]
        train_df = df[~df["signer_id"].isin(train_not_unique_signer_ids)]
        val_df = pd.concat([df[df["signer_id"].isin(val_unique_signer_ids)], val_df], ignore_index=True)
        test_df = pd.concat([df[df["signer_id"].isin(test_unique_signer_ids)], test_df], ignore_index=True)
        for signer_id, (train_view, val_view, test_view) in common_signer_ids.items():
            if train_view in view_ids:
                train_df = pd.concat([df[(df["signer_id"] == signer_id) & (df["view"] == train_view)], train_df],
                                     ignore_index=True)
            if val_view in view_ids:
                val_df = pd.concat([df[(df["signer_id"] == signer_id) & (df["view"] == val_view)], val_df],
                                   ignore_index=True)
            if test_view in view_ids:
                test_df = pd.concat([df[(df["signer_id"] == signer_id) & (df["view"] == test_view)], test_df],
                                    ignore_index=True)
        if self.split == "train":
            final_df = train_df
        elif self.split == "val":
            final_df = val_df
        elif self.split == "test":
            final_df = test_df
        else:
            raise ValueError("Split is 'train', 'val' or 'test'")

        final_df = final_df.dropna(subset=['gloss_id', 'pose_path'])
        final_df['gloss_id'] = final_df['gloss_id'].astype(int)
        return final_df.reset_index(drop=True)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        row = self.data_df.iloc[index]

        with open(row['pose_path'], "rb") as f:
            data = f.read()
            try:
                pose_obj = Pose.read(data)
            except Exception as e:
                print(f"Bad file: {row['pose_path']}, size={len(data)}")
                raise e

        pose_data = self.point_selector(pose_obj)

        if self.transform:
            pose_data = self.transform(pose_data)

        pose_data = pose_data.reshape(pose_data.size(0), -1)
        label = torch.tensor(row['gloss_id'], dtype=torch.long)

        return pose_data, label

if __name__ == '__main__':
    train = VSLPoseDataset(root_dir='../dataset/root', split='train', transform=train_transforms)
    test = VSLPoseDataset(root_dir='../dataset/root', split='test', transform=val_test_transforms)
    val = VSLPoseDataset(root_dir='../dataset/root', split='val', transform=val_test_transforms)
    len_train = len(train)
    len_val = len(val)
    len_test = len(test)
    sum = len_train + len_val + len_test
    print(f'Total videos: {sum}')
    print(f'Train videos: {len_train}, Percent Train: {len_train/sum*100}%')
    print(f'Val videos: {len_val}, Percent Val: {len_val/sum*100}%')
    print(f'Test videos: {len_test}, Percent Test: {len_test/sum*100}%')

    pose, label = train.__getitem__(3)
    print(pose.size())

