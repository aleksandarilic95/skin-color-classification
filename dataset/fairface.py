from torch.utils.data import Dataset, DataLoader
import torch

import os
from PIL import Image
import pandas as pd

class FairFaceDataset(Dataset):
    def __init__(self, root, csv, transform = None):
        self.root = root
        self.df = pd.read_csv(os.path.join(self.root, csv))
        self.transforms = transform

        self.mapping = {
            'White': 0, 
            'Black': 1, 
            'Latino_Hispanic': 2, 
            'East Asian': 3, 
            'Southeast Asian': 3, 
            'Indian': 4, 
            'Middle Eastern': 5
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'file']
        img = Image.open(os.path.join(self.root, img_name))

        if self.transforms is not None:
            img = self.transforms(img)

        target = self.df.loc[idx, 'race']
        target = self.mapping[target]
        target = torch.tensor(target, dtype = torch.int64)

        return img, target

def get_fairface_trainval(config,
                       transform = None,
                       target_transform = None):
    result_list = []

    for split in ['train', 'val']:
        dataset = get_dataset(config['ROOT'], f'{split}.csv', transform[split])
        loader = get_dataloader(dataset, config['BATCH_SIZE'], config['SHUFFLE'], config['NUM_WORKERS'])
        result_list.append(loader)

    return {'train': result_list[0],
            'val': result_list[1]}

def get_dataloader(dataset,
                   batch_size = 1,
                   shuffle = False,
                   num_workers = 0):
    return DataLoader(dataset = dataset,
                      batch_size = batch_size,
                      shuffle = shuffle,
                      num_workers = num_workers,
                      drop_last = True
    )

def get_dataset(root,
                csv,
                transform = None
                ):
    return FairFaceDataset(
        root = root,
        csv = csv,
        transform = transform
    )