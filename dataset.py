import os
import numpy
import torch
from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):
    def __init__(self, data_path):
        self.dataset = []
        self.labels = []

        numpy_dir = os.path.join(data_path, 'numpy')
        for filename in os.listdir(numpy_dir):
            if filename.endswith('.npy'):
                label_id = int(filename.split('.npy')[0])
                file_path = os.path.join(numpy_dir, filename)
                class_data = numpy.load(file_path)

                self.dataset.extend(class_data)
                self.labels.extend([label_id] * len(class_data))

        self.dataset = torch.tensor(numpy.array(self.dataset), dtype=torch.float32)
        self.labels = torch.tensor(numpy.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

