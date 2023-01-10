from torch.utils.data import Dataset
import os
from glob import glob
import random
import torchvision as tv
import torchvision.transforms.functional as TF
from .transforms import *


class TrainDAVIS(Dataset):
    def __init__(self, root, year, split, clip_n):
        self.root = root
        self.clip_n = clip_n
        with open(os.path.join(root, 'ImageSets', '{}/{}.txt'.format(year, split)), 'r') as f:
            self.video_list = f.read().splitlines()
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

        # get flip param
        h_flip = False
        if random.random() > 0.5:
            h_flip = True
        v_flip = False
        if random.random() > 0.5:
            v_flip = True

        # select training frames
        all_frames = list(range(len(img_list)))
        selected_frames = random.sample(all_frames, 1)

        img_lst = []
        flow_lst = []
        mask_lst = []
        for i, frame_id in enumerate(selected_frames):
            img = load_image_in_PIL(img_list[frame_id], 'RGB')
            flow = load_image_in_PIL(flow_list[frame_id], 'RGB')
            mask = load_image_in_PIL(mask_list[frame_id], 'P')

            # resize to 384p
            img = img.resize((384, 384), Image.BICUBIC)
            flow = flow.resize((384, 384), Image.BICUBIC)
            mask = mask.resize((384, 384), Image.NEAREST)

            # joint flip
            if h_flip:
                img = TF.hflip(img)
                flow = TF.hflip(flow)
                mask = TF.hflip(mask)
            if v_flip:
                img = TF.vflip(img)
                flow = TF.vflip(flow)
                mask = TF.vflip(mask)

            img_lst.append(self.to_tensor(img))
            flow_lst.append(self.to_tensor(flow))
            mask_lst.append(self.to_mask(mask))

        imgs = torch.stack(img_lst, 0)
        flows = torch.stack(flow_lst, 0)
        masks = torch.stack(mask_lst, 0)
        masks = (masks != 0).long()
        return {'imgs': imgs, 'flows': flows, 'masks': masks}


class TestDAVIS(Dataset):
    def __init__(self, root, year, split):
        self.root = root
        self.year = year
        self.split = split
        self.init_data()

    def read_img(self, path):
        pic = Image.open(path)
        transform = tv.transforms.ToTensor()
        return transform(pic)

    def read_mask(self, path):
        pic = Image.open(path)
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
        given_masks = [self.read_mask(os.path.join(mask_path, '{:05d}.png'.format(0))).unsqueeze(0)] + [None] * (len(frame_ids) - 1)
        files = ['{:05d}.png'.format(i) for i in frame_ids]
        if self.split == 'test-dev':
            return {'imgs': imgs, 'flows': flows, 'given_masks': given_masks, 'files': files, 'val_frame_ids': None}
        masks = torch.stack([self.read_mask(os.path.join(mask_path, '{:05d}.png'.format(i))) for i in frame_ids]).squeeze().unsqueeze(0)
        if self.year == '2016':
            masks = (masks != 0).long()
            given_masks[0] = (given_masks[0] != 0).long()
        return {'imgs': imgs, 'flows': flows, 'given_masks': given_masks, 'masks': masks, 'files': files, 'val_frame_ids': None}

    def get_video(self, video_name):
        frame_ids = sorted([int(os.path.splitext(file)[0]) for file in os.listdir(os.path.join(self.root, 'JPEGImages', '480p', video_name))])
        yield self.get_snippet(video_name, frame_ids)

    def get_videos(self):
        for video_name in self.video_list:
            yield video_name, self.get_video(video_name)
