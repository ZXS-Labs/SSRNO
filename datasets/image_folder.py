import os
import json
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchio as tio
import torch
from datasets import register
from tqdm import tqdm
import matplotlib.pyplot as plt

@register('IXI-processed')
class IXI_Processed(Dataset):
    def __init__(self, root_path, gt_path):
        lrfiles = sorted(os.listdir(root_path))
        hrfiles = sorted(os.listdir(gt_path))
        self.lrs = []
        self.hrs = []
        self.dtr = []
        for file in hrfiles:
            tmp = torch.Tensor(np.load(os.path.join(gt_path,file))).permute(2,0,1)
            assert tmp.shape[0]==96
            self.dtr.append(torch.max(tmp))
            self.hrs.append(tmp/torch.max(tmp))
        self.hrs = torch.cat(self.hrs)
        cnt = 0
        for file in lrfiles: 
            tmp = torch.Tensor(np.load(os.path.join(root_path,file))).permute(2,0,1)
            self.lrs.append(tmp/self.dtr[cnt])
            cnt = cnt + 1
        self.lrs = torch.cat(self.lrs)
    def __len__(self):
        return len(self.lrs)

    def __getitem__(self, idx):
        lr = self.lrs[idx]
        hr = self.hrs[idx]
        return {
            "lr": lr,
            "hr": hr
        }

@register('IXI')
class IXI(Dataset):
    def __init__(self, root_path):
        transforms = [
            tio.ToCanonical(),  # to RAS
            tio.Resample((1, 1, 1)),  # to 1 mm iso
        ]
        ixi_dataset = tio.datasets.IXI(
            root_path,
            modalities=('T2',),
            transform=tio.Compose(transforms),
            download=True,
        )
        print('Number of subjects in dataset:', len(ixi_dataset))
        self.dataset = []
        self.mx = []
        for i in tqdm(range(70)):
            now = ixi_dataset[i+500]['T2']['data'][0].permute(2,0,1)
            nowmx = torch.max(now)
            for j in now:
                self.mx.append(nowmx)
                self.dataset.append(j/nowmx)
        print("len of dataset:", len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]
        return x


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1):
        self.repeat = repeat

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            self.files.append(transforms.ToTensor()(
                Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        return x


