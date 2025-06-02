from .transforms import *
import os
from glob import glob
from PIL import Image
import torchvision as tv


class TestYTOBJ(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.video_list = []
        class_list = sorted(os.listdir(os.path.join(root, 'JPEGImages')))
        for class_name in class_list:
            video_list = sorted(os.listdir(os.path.join(root, 'JPEGImages', class_name)))
            for video_name in video_list:
                self.video_list.append(class_name + '_' + video_name)
        self.to_tensor = tv.transforms.ToTensor()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        class_name = self.video_list[idx].split('_')[0]
        video_name = self.video_list[idx].split('_')[1]
        img_dir = os.path.join(self.root, 'JPEGImages', class_name, video_name)
        flow_dir = os.path.join(self.root, 'JPEGFlows', class_name, video_name)
        mask_dir = os.path.join(self.root, 'Annotations', class_name, video_name)
        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        flow_list = sorted(glob(os.path.join(flow_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        # generate testing snippets
        imgs = []
        flows = []
        masks = []
        for i in range(len(img_list)):
            img = Image.open(img_list[i]).convert('RGB')
            imgs.append(self.to_tensor(img))
        for i in range(len(flow_list)):
            flow = Image.open(flow_list[i]).convert('RGB')
            flows.append(self.to_tensor(flow))
        for i in range(len(mask_list)):
            mask = Image.open(mask_list[i]).convert('L')
            masks.append(self.to_tensor(mask))

        # gather all frames
        imgs = torch.stack(imgs, dim=0)
        flows = torch.stack(flows, dim=0)
        masks = torch.stack(masks, dim=0)
        masks = (masks > 0.5).long()
        return {'imgs': imgs, 'flows': flows, 'masks': masks, 'class_name': class_name, 'video_name': video_name, 'files': mask_list}
