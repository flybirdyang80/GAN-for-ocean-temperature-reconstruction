import glob
import random
import os
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import re


def get_month_by_division(day_number):
    day_number = re.findall(r'\d+', day_number)
    day_number = int(day_number[0]) if day_number else None
    day_number = int(day_number)
    days_in_year = day_number % 360
    if days_in_year == 0:
        return 12
    month = (days_in_year - 1) // 30 + 1  # 减1是为了让1-30天算作第1个月，31-60天算作第2个月，依此类推

    return month


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):
        # Load .pt file
        data = torch.load(self.files[index % len(self.files)])
        filename = os.path.basename(self.files[index % len(self.files)])
        # Split the data
        img_A = data[:, :, :2]
        img_B = data[:, :, 2:4]
        Argo_data = data[:, :, 2:4]
        # img_B = img_B.unsqueeze(2)
        # img_B = img_B.repeat(1, 1, 3)

        img_A = torch.nan_to_num(img_A)
        img_B = torch.nan_to_num(img_B)
        Argo_data = torch.nan_to_num(Argo_data)
        # for i in range(img_A.shape[2]):
        #     img_A[:, :, i] = (img_A[:, :, i] - np.mean(img_A[:, :, i])) / np.std(img_A[:, :, i])

        # img_A = torch.from_numpy(img_A).float().unsqueeze(0)
        img_A = img_A.permute(2, 0, 1)
        # img_B = img_B.unsqueeze(0)
        img_B = img_B.permute(2, 0, 1)
        # img_B = img_B.transpose(2, 0, 1)
        Argo_data = Argo_data.permute(2, 0, 1)
        month = get_month_by_division(filename)
        # 确定是哪一天
        day_number = re.findall(r'\d+', filename)
        day_number = int(day_number[0]) if day_number else None
        day_number = int(day_number)
        season_label = np.zeros((12,))
        season_label[int(month) - 1] = 1

        return {"A": img_A, "B": img_B, "Argo": Argo_data,  "season_label": season_label, "day_number": day_number}

    def __len__(self):
        return len(self.files)
