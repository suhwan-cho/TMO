from .transforms import *
import os
import random
from glob import glob
from PIL import Image
import torchvision as tv
import torchvision.transforms.functional as TF


class TrainDAVIS(torch.utils.data.Dataset):
    def __init__(self, root, year, split, clip_n):
        self.root = root
        with open(os.path.join(root, 'ImageSets', '{}/{}.txt'.format(year, split)), 'r') as f:
            self.video_list = f.read().splitlines()
        self.clip_n = clip_n
        self.to_tensor = tv.transforms.ToTensor()
        self.to_mask = LabelToLongTensor()

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):
        video_name = random.choice(self.video_list)
        img_dir = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        flow_dir = os.path.join(self.root, 'JPEGFlows', '480p', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', '480p', video_name)
        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        flow_list = sorted(glob(os.path.join(flow_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        # select training frame
        all_frames = list(range(len(img_list)))
        frame_id = random.choice(all_frames)
        img = Image.open(img_list[frame_id]).convert('RGB')
        flow = Image.open(flow_list[frame_id]).convert('RGB')
        mask = Image.open(mask_list[frame_id]).convert('P')

        # resize to 384p
        img = img.resize((384, 384), Image.BICUBIC)
        flow = flow.resize((384, 384), Image.BICUBIC)
        mask = mask.resize((384, 384), Image.NEAREST)

        # joint flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            flow = TF.hflip(flow)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            flow = TF.vflip(flow)
            mask = TF.vflip(mask)

        # convert formats
        imgs = self.to_tensor(img).unsqueeze(0)
        flows = self.to_tensor(flow).unsqueeze(0)
        indices = torch.ones(1, 1, 1, 1)
        masks = self.to_mask(mask).unsqueeze(0)
        masks = (masks != 0).long()
        return {'imgs': imgs, 'flows': flows, 'indices': indices, 'masks': masks}


class TestDAVIS(torch.utils.data.Dataset):
    def __init__(self, root, year, split):
        self.root = root
        self.year = year
        self.split = split
        self.init_data()

    def read_img(self, path):
        pic = Image.open(path).convert('RGB')
        transform = tv.transforms.ToTensor()
        return transform(pic)

    def read_mask(self, path):
        pic = Image.open(path).convert('P')
        transform = LabelToLongTensor()
        return transform(pic)

    def init_data(self):
        with open(os.path.join(self.root, 'ImageSets', self.year, self.split + '.txt'), 'r') as f:
            self.video_list = sorted(f.read().splitlines())
            print('--- DAVIS {} {} loaded for testing ---'.format(self.year, self.split))

    def get_snippet(self, video_name, frame_ids):
        img_path = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        flow_path = os.path.join(self.root, 'JPEGFlows', '480p', video_name)
        mask_path = os.path.join(self.root, 'Annotations', '480p', video_name)
        imgs = torch.stack([self.read_img(os.path.join(img_path, '{:05d}.jpg'.format(i))) for i in frame_ids]).unsqueeze(0)
        flows = torch.stack([self.read_img(os.path.join(flow_path, '{:05d}.jpg'.format(i))) for i in frame_ids]).unsqueeze(0)
        masks = torch.stack([self.read_mask(os.path.join(mask_path, '{:05d}.png'.format(i))) for i in frame_ids]).unsqueeze(0)
        if self.year == '2016':
            masks = (masks != 0).long()
        files = ['{:05d}.png'.format(i) for i in frame_ids]
        return {'imgs': imgs, 'flows': flows, 'masks': masks, 'files': files}

    def get_video(self, video_name):
        frame_ids = sorted([int(file[:5]) for file in os.listdir(os.path.join(self.root, 'JPEGImages', '480p', video_name))])
        yield self.get_snippet(video_name, frame_ids)

    def get_videos(self):
        for video_name in self.video_list:
            yield video_name, self.get_video(video_name)
