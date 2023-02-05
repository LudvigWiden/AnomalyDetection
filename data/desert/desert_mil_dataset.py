import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms, utils
import matplotlib.patches as patches

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def show_image(image):
    plt.imshow(image)
    plt.show()

class DesertSequenceDataset(Dataset):
    
    def __init__(self, anomaly=False, train=True, seq_len = 16, num_train_seq=4):
        """
        Args:
            anomaly (bool): True if anomaly images, False if normal images
            transform (callable, optional): Optional transform to be applied
        """
        self.anomaly = anomaly
        self.train = train
        self.num_train_seq = num_train_seq
        self.seq_len = seq_len
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        if self.anomaly:
            self.image_path = os.path.join(self.base_path,"anomaly")
        else:
            self.image_path = os.path.join(self.base_path,"normal")

        self.image_folders = os.listdir(self.image_path)
        
        self.image_folders = [f for f in self.image_folders if os.path.isdir(os.path.join(self.image_path, f))]
        if self.train:
            self.image_folders = self.image_folders[:-self.num_train_seq]
        else:
            self.image_folders = self.image_folders[-self.num_train_seq:]
        print(self.image_folders)

        num_files = [len(os.listdir(os.path.join(self.image_path, f))) for f in self.image_folders]

        file_idx = {}
        self.num_files = 0
        for i,f in enumerate(self.image_folders):
            file_idx[self.num_files] = f
            self.num_files += num_files[i] - self.seq_len
        self.file_idx = file_idx

    def transform(self, images):
        # Transform to tensor
        for i in range(len(images)):
            images[i] = TF.to_tensor(images[i]) 
        
        x, y, h, w = transforms.RandomCrop.get_params(
            images[0], output_size=(112, 112))
        for i in range(len(images)):
            images[i] = TF.crop(images[i], x, y, h, w)
        
        return images

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        prev_key = 0
        folder = None
        for i in self.file_idx:
            if idx < i:
                folder = self.file_idx[prev_key]
                break
            prev_key = i
        if folder == None:
            folder = self.file_idx[prev_key]
        images = []
        for i in range(self.seq_len):
            image_name = os.path.join(self.image_path, folder, str(idx + i - prev_key + 1)+".png")
            image = io.imread(image_name)
            images.append(image)
        
        images = self.transform(images)
        images = torch.stack(images)
        
        if self.train:
            return images
        else:
            return images, folder


if __name__ == "__main__":
    ds = DesertSequenceDataset(anomaly=True, train=True, seq_len=16, num_train_seq=3)
    dsn = DesertSequenceDataset(anomaly=False, train=True, seq_len=16, num_train_seq=3)
    print(len(ds))

    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    dl2 = DataLoader(dsn, batch_size=1, shuffle=True, num_workers=0)
    for i, (data1,data2) in enumerate(zip(dl,dl2)):
        print(data1.shape, data2.shape)
        break
