import torchvision as tv
import torch
import torch.nn as nn
import torch.nn.functional as F


# basic modules
class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(ConvRelu(c, c, 1, 1, 0), Conv(c, c, 1, 1, 0))
        self.conv3 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        c = torch.sigmoid(self.conv2(F.adaptive_avg_pool2d(x, output_size=(1, 1))) + self.conv2(F.adaptive_max_pool2d(x, output_size=(1, 1))))
        x = x * c
        s = torch.sigmoid(self.conv3(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))
        x = x * s
        return x


# encoding module
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = tv.models.resnet101(pretrained=True)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img):
        x = (img - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        s4 = x
        x = self.layer2(x)
        s8 = x
        x = self.layer3(x)
        s16 = x
        x = self.layer4(x)
        s32 = x
        return {'s32': s32, 's16': s16, 's8': s8, 's4': s4}


# decoding module
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvRelu(2048, 256, 1, 1, 0)
        self.blend1 = ConvRelu(256, 256, 3, 1, 1)
        self.cbam1 = CBAM(256)
        self.conv2 = ConvRelu(1024, 256, 1, 1, 0)
        self.blend2 = ConvRelu(256 + 256, 256, 3, 1, 1)
        self.cbam2 = CBAM(256)
        self.conv3 = ConvRelu(512, 256, 1, 1, 0)
        self.blend3 = ConvRelu(256 + 256, 256, 3, 1, 1)
        self.cbam3 = CBAM(256)
        self.conv4 = ConvRelu(256, 256, 1, 1, 0)
        self.blend4 = ConvRelu(256 + 256, 256, 3, 1, 1)
        self.cbam4 = CBAM(256)
        self.predictor = Conv(256, 2, 3, 1, 1)

    def forward(self, img_feats, flow_feats):
        x = self.conv1(img_feats['s32'] + flow_feats['s32'])
        x = self.cbam1(self.blend1(x))
        s16 = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = torch.cat([self.conv2(img_feats['s16'] + flow_feats['s16']), s16], dim=1)
        x = self.cbam2(self.blend2(x))
        s8 = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = torch.cat([self.conv3(img_feats['s8'] + flow_feats['s8']), s8], dim=1)
        x = self.cbam3(self.blend3(x))
        s4 = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = torch.cat([self.conv4(img_feats['s4'] + flow_feats['s4']), s4], dim=1)
        x = self.predictor(self.cbam4(self.blend4(x)))
        x = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        return x


# VOS model
class VOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = Encoder()
        self.flow_encoder = Encoder()
        self.decoder = Decoder()


# TMO model
class TMO(nn.Module):
    def __init__(self):
        super().__init__()
        self.vos = VOS()

    def forward(self, imgs, flows):
        B, L, _, H1, W1 = imgs.size()
        _, _, _, H2, W2 = flows.size()

        # resize to 384p
        s = 384
        if (H1, W1) != (s, s) or (H2, W2) != (s, s):
            imgs = F.interpolate(imgs.view(B * L, -1, H1, W1), size=(s, s), mode='bicubic', align_corners=False).view(B, L, -1, s, s)
            flows = F.interpolate(flows.view(B * L, -1, H2, W2), size=(s, s), mode='bicubic', align_corners=False).view(B, L, -1, s, s)

        # for each frame
        score_lst = []
        mask_lst = []
        for i in range(0, L):

            # query frame prediction
            img_feats = self.vos.img_encoder(imgs[:, i])
            flow_feats = self.vos.flow_encoder(flows[:, i])
            final_score = self.vos.decoder(img_feats, flow_feats)
            if (H1, W1) != (s, s):
                final_score = F.interpolate(final_score, size=(H1, W1), mode='bicubic', align_corners=False)

            # generate soft scores
            if B != 1:
                score_lst.append(final_score)

            # generate hard masks
            if B == 1:
                pred_seg = torch.softmax(final_score, dim=1)
                pred_mask = torch.max(pred_seg, dim=1, keepdim=True)[1]
                mask_lst.append(pred_mask)

        # generate output
        output = {}
        if B != 1:
            output['scores'] = torch.stack(score_lst, dim=1)
        if B == 1:
            output['masks'] = torch.stack(mask_lst, dim=1)
        return output
