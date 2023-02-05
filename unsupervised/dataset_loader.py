from torchvision.datasets import MNIST, CIFAR10,ImageFolder
from torchvision import transforms
from torch.utils.data import Subset,ConcatDataset,Dataset,DataLoader
import torch
import numpy as np
import os
import os
import glob
from pathlib import Path
from PIL import Image

class MySampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):        
        indices = []
        for i in range(len(end_idx)-1):
            start = end_idx[i]
            end = end_idx[i+1] - seq_length
            indices.append(torch.arange(start, end))
        indices = torch.cat(indices)
        self.indices = indices
        
    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.indices)


class SeqDataset(Dataset):
    def __init__(self, image_paths, seq_length, transform, length):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.labels = [ip[1] for ip in self.image_paths]

        
    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        #print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))
        images = []
        for i in indices:
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        x = torch.stack(images)
        y = torch.tensor(self.labels[start], dtype=torch.long)
        
        return x, y
    

# Load normal data from local folder 'normal_path', and anomaly data from
# local folder 'anomaly_path'. Files are loaded into batches
# with frames in sequence to represent videos. Returns train, validation
# test data
def load_data(normal_path, anomaly_path, seq_length):

    train_image_paths = []
    test_image_paths = []
    val_image_paths = []
    train_end_idx = []
    test_end_idx = []
    val_end_idx = []
    
    #### NORMAL VIDEOS ####
    # Get the number of normal movies
    folder = glob.glob(normal_path+"/*")
    nr_of_normal_videos = len(folder)
    
    # Pick 2 normal videos for validation and test
    random_videos = np.random.choice(range(1,nr_of_normal_videos+1), 4, replace=False)
    val_video = random_videos[0:2]
    test_video = random_videos[2:4]

    i = 1
    print(f"Loading {nr_of_normal_videos} normal movies")
    for d in os.scandir(normal_path):
        if d.is_dir:
            paths = sorted(glob.glob(os.path.join(d.path, '*.png')))

            # Add class idx to paths
            paths = [(p, 1) for p in paths]
            # Use all normal data for training
            
            train_image_paths.extend(paths)
            train_end_idx.extend([len(paths)])

            # 1 Video for test
            if i in test_video:
                test_image_paths.extend(paths)
                test_end_idx.extend([len(paths)])
            # 1 Video for validation
            elif  i in val_video:
                val_image_paths.extend(paths)
                val_end_idx.extend([len(paths)])
            i+=1
        
    print(f"{len(train_end_idx)} normal videos to train")
    print(f"{len(val_end_idx)} normal videos to validation")
    print(f"{len(test_end_idx)} normal videos to test")

    #### ANOMALY VIDEOS ####
    folder = glob.glob(anomaly_path+"/*")
    nr_of_anomaly_movies = len(folder)
    # Use 50% for validation and 50% for test
    test_portion = int(np.floor(0.5 * nr_of_anomaly_movies))
    test_classes = np.random.choice(range(1, nr_of_anomaly_movies+1), test_portion, replace=False)
    print(f"Loading {nr_of_anomaly_movies} anomaly movies")
    i = 1
    for d in os.scandir(anomaly_path):
        if d.is_dir:
            paths = sorted(glob.glob(os.path.join(d.path, '*.png')))
            # Add class idx to paths
            paths = [(p, 0) for p in paths]
            
            if i in test_classes:
                test_image_paths.extend(paths)
                test_end_idx.extend([len(paths)])
                
            else:
                val_image_paths.extend(paths)
                val_end_idx.extend([len(paths)])
            i+=1


    print(f"{len(val_end_idx)} normal + anomaly videos to validation")
    print(f"{len(test_end_idx)} normal + anomaly videos to test")
    
    # Number of frames per video
    train_end_idx = [0, *train_end_idx]
    val_end_idx = [0, *val_end_idx]
    test_end_idx = [0, *test_end_idx]
    
    # summed
    train_end_idx = torch.cumsum(torch.tensor(train_end_idx), 0)
    val_end_idx = torch.cumsum(torch.tensor(val_end_idx), 0)
    test_end_idx = torch.cumsum(torch.tensor(test_end_idx), 0)

    test_sampler = MySampler(test_end_idx, seq_length)
    train_sampler = MySampler(train_end_idx, seq_length)
    val_sampler = MySampler(val_end_idx, seq_length)

    transform = transforms.Compose([
        transforms.Resize((124, 124)),
        transforms.ToTensor()
    ])
    
    train_dataset = SeqDataset(
        image_paths=train_image_paths,
        seq_length=seq_length,
        transform=transform,
        length=len(train_sampler))

    val_dataset = SeqDataset(
        image_paths=val_image_paths,
        seq_length=seq_length,
        transform=transform,
        length=len(val_sampler))
    
    test_dataset = SeqDataset(
        image_paths=test_image_paths,
        seq_length=seq_length,
        transform=transform,
        length=len(test_sampler))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler
    )
    
    print(f"Loaded {len(train_loader)} train sequence batches")
    print(f"Loaded {len(val_loader)}  validation sequence batches")
    print(f"Loaded {len(test_loader)} test sequence batches")

    
    return train_loader, val_loader, test_loader


# Load normal data from local folder 'normal_path', and anomaly data from
# local folder 'anomaly_path'. Shape images to size (image_dim, image_dim)
# and return train, validation and anomaly data
def load_dataset(normal_path, anomaly_path, image_dim, batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_dim,image_dim)),
            transforms.ToTensor(),
        ])

    normal_dataset = ImageFolder(normal_path, transform=transform)
    anomaly_dataset = ImageFolder(anomaly_path, transform=transform)

    # NORMAL DATA
    normal_classes = list(normal_dataset.class_to_idx.values())
    number_of_normal_classes = len(normal_classes)
    
    # 80% dedicated to train
    train_portion = int(np.floor(0.8 * number_of_normal_classes))
    
    # Randomize video labels for train data
    train_classes = np.random.choice(normal_classes, train_portion, replace=False)
    remaining_classes = [idx for idx in normal_classes if idx not in train_classes]
    number_of_remaining_classes = len(remaining_classes)
    test_portion = int(np.floor(0.5 * number_of_remaining_classes))
    test_classes = np.random.choice(remaining_classes, test_portion, replace=False)

    # Get image indices for train and validation/test
    train_indices = []
    test_indices = []
    val_indices = []

    # Divide normal data (80%, 10%, 10%)
    for i, label in enumerate(normal_dataset.targets):
        if label in train_classes:
            train_indices.append(i)
        elif label in test_classes:
            test_indices.append(i)
        else:
            val_indices.append(i)
    
    # Train dataset
    train_set = Subset(normal_dataset, train_indices)
    # Portion of normal data for test and validation data
    val_normal = Subset(normal_dataset, val_indices)
    test_normal = Subset(normal_dataset, test_indices)
    
    # ANOMALY DATA
    anomaly_classes = list(anomaly_dataset.class_to_idx.values())
    number_of_anomaly_classes = len(anomaly_classes)

    # 50% dedicated to test, 50% to validation
    test_portion = int(np.floor(0.5 * number_of_anomaly_classes))
    test_classes = np.random.choice(anomaly_classes, test_portion, replace=False)
    # Divide normal data (80%, 10%, 10%)
    test_indices = []
    val_indices = []
    for i, label in enumerate(anomaly_dataset.targets):
        if label in test_classes:
            test_indices.append(i)
        else:
            val_indices.append(i)
    # Portion of normal data for test and validation data
    val_anomaly = Subset(anomaly_dataset, val_indices)
    test_anomaly = Subset(anomaly_dataset, test_indices)
    
    # Test data with normal + anomalies
    test_set = ConcatDataset([test_normal, test_anomaly])
    val_set = ConcatDataset([val_normal, val_anomaly])

    print(f"Loaded {len(train_set)} training images")
    print(f"Loaded {len(val_set)} validation images")
    print(f"Loaded {len(test_set) } test images")
    
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=2)
    validation_loader = DataLoader(val_set, batch_size=batch_size,shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=True, num_workers=2)
    
    return train_loader, validation_loader, val_normal, val_anomaly, test_loader, test_normal, test_anomaly