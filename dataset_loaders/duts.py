from torch.utils.data import Dataset
import os
from glob import glob
import random
import torchvision as tv
import torchvision.transforms.functional as TF
from .transforms import *


class TrainDUTS(Dataset):
    def __init__(self, root, clip_n):
        self.root = root
        self.clip_n = clip_n
        img_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'Annotations')
        self.img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        self.mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))
        self.to_tensor = tv.transforms.ToTensor()
        self.to_mask = LabelToLongTensor()

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):
        all_frames = list(range(len(self.img_list)))
        k = random.choice(all_frames)
        img = load_image_in_PIL(self.img_list[k], 'RGB')
        mask = load_image_in_PIL(self.mask_list[k], 'P')

        # resize to 384p
        img = img.resize((384, 384), Image.BICUBIC)
        mask = mask.resize((384, 384), Image.NEAREST)

        # joint flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        imgs = self.to_tensor(img).unsqueeze(0)
        masks = self.to_mask(mask).unsqueeze(0)
        return {'imgs': imgs, 'flows': imgs, 'masks': masks}
