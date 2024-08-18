from evaluation import metrics
from utils import AverageMeter, get_iou
import copy
import numpy
import torch


class Trainer(object):
    def __init__(self, model, ver, optimizer, train_loader, val_set, save_name, save_step, val_step):
        self.model = model.cuda()
        self.ver = ver
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_set = val_set
        self.save_name = save_name
        self.save_step = save_step
        self.val_step = val_step
        self.epoch = 1
        self.best_score = 0
        self.score = 0
        self.stats = {'loss': AverageMeter(), 'iou': AverageMeter()}

    def train(self, max_epochs):
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_epoch()
            if self.epoch % self.save_step == 0:
                print('saving checkpoint\n')
                self.save_checkpoint()
            if self.score > self.best_score:
                print('new best checkpoint, after epoch {}\n'.format(self.epoch))
                self.save_checkpoint(alt_name='best')
                self.best_score = self.score
        print('finished training!\n', flush=True)

    def train_epoch(self):

        # train
        if self.ver != 'rn101':
            self.model.train()
        self.cycle_dataset(mode='train')

        # val
        if self.ver != 'rn101':
            self.model.eval()
        if self.epoch % self.val_step == 0:
            if self.val_set is not None:
                with torch.no_grad():
                    self.score = self.cycle_dataset(mode='val')

        # update stats
        for stat_value in self.stats.values():
            stat_value.new_epoch()

    def cycle_dataset(self, mode):
        if mode == 'train':
            for vos_data in self.train_loader:
                imgs = vos_data['imgs'].cuda()
                flows = vos_data['flows'].cuda()
                indices = vos_data['indices'].cuda()
                masks = vos_data['masks'].cuda()
                B, L, _, H, W = imgs.size()

                # embed motion validity index
                flows = indices * flows + (1 - indices) * imgs

                # model run
                vos_out = self.model(imgs, flows)
                loss = torch.nn.CrossEntropyLoss()(vos_out['scores'].view(B * L, 2, H, W), masks.reshape(B * L, H, W))

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # loss, iou
                self.stats['loss'].update(loss.detach().cpu().item(), B)
                iou = torch.mean(get_iou(vos_out['scores'].view(B * L, 2, H, W), masks.reshape(B * L, H, W))[:, 1:])
                self.stats['iou'].update(iou.detach().cpu().item(), B)

            print('[ep{:04d}] loss: {:.5f}, iou: {:.5f}'.format(self.epoch, self.stats['loss'].avg, self.stats['iou'].avg))

        if mode == 'val':
            metrics_res = {}
            metrics_res['J'] = []
            metrics_res['F'] = []
            for video_name, video_parts in self.val_set.get_videos():
                for vos_data in video_parts:
                    imgs = vos_data['imgs'].cuda()
                    flows = vos_data['flows'].cuda()
                    masks = vos_data['masks'].cuda()

                    # inference
                    vos_out = self.model(imgs, flows)
                    res_masks = vos_out['masks'][:, 1:-1].squeeze(2)
                    gt_masks = masks[:, 1:-1].squeeze(2)
                    B, L, H, W = res_masks.shape
                    object_ids = numpy.unique(gt_masks.cpu()).tolist()
                    object_ids.remove(0)

                    # evaluate output
                    all_res_masks = numpy.zeros((len(object_ids), L, H, W))
                    all_gt_masks = numpy.zeros((len(object_ids), L, H, W))
                    for k in object_ids:
                        res_masks_k = copy.deepcopy(res_masks).cpu().numpy()
                        res_masks_k[res_masks_k != k] = 0
                        res_masks_k[res_masks_k != 0] = 1
                        all_res_masks[k - 1] = res_masks_k[0]
                        gt_masks_k = copy.deepcopy(gt_masks).cpu().numpy()
                        gt_masks_k[gt_masks_k != k] = 0
                        gt_masks_k[gt_masks_k != 0] = 1
                        all_gt_masks[k - 1] = gt_masks_k[0]

                    # calculate scores
                    j_metrics_res = numpy.zeros(all_gt_masks.shape[:2])
                    f_metrics_res = numpy.zeros(all_gt_masks.shape[:2])
                    for i in range(all_gt_masks.shape[0]):
                        j_metrics_res[i] = metrics.db_eval_iou(all_gt_masks[i], all_res_masks[i])
                        f_metrics_res[i] = metrics.db_eval_boundary(all_gt_masks[i], all_res_masks[i])
                        [JM, _, _] = metrics.db_statistics(j_metrics_res[i])
                        metrics_res['J'].append(JM)
                        [FM, _, _] = metrics.db_statistics(f_metrics_res[i])
                        metrics_res['F'].append(FM)

            # gather scores
            J, F = metrics_res['J'], metrics_res['F']
            final_mean = (numpy.mean(J) + numpy.mean(F)) / 2.
            print('[ep{:04d}] J&F score: {:.5f}\n'.format(self.epoch, final_mean))
            return final_mean

    def save_checkpoint(self, alt_name=None):
        if alt_name is not None:
            file_path = 'weights/{}_{}.pth'.format(self.save_name, alt_name)
        else:
            file_path = 'weights/{}_{:04d}.pth'.format(self.save_name, self.epoch)
        torch.save(self.model.module.state_dict(), file_path)
