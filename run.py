from torch.utils.data import DataLoader
from dataset_loaders import *
import evaluation
from tmo import TMO
from trainer import Trainer
from optparse import OptionParser
import warnings
warnings.filterwarnings('ignore')


parser = OptionParser()
parser.add_option('--train', action='store_true', dest='train', default=None)
parser.add_option('--test', action='store_true', dest='test', default=None)
(options, args) = parser.parse_args()

torch.cuda.set_device(0)


##################
# Train
##################
def train_duts_davis(model):
    duts_set = TrainDUTS('../DB/DUTS', clip_n=384)
    davis_set = TrainDAVIS('../DB/DAVIS', '2016', 'train', clip_n=128)
    train_set = torch.utils.data.ConcatDataset([duts_set, davis_set])
    train_loader = DataLoader(train_set, shuffle=True, batch_size=16, num_workers=4)
    val_set = TestDAVIS('../DB/DAVIS', '2016', 'val')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    trainer = Trainer(model, optimizer, train_loader, val_set, save_name='duts_davis', save_step=1000, val_step=100)
    trainer.train(4000)


##################
# Test
##################
def test_davis(model):
    datasets = {
        'DAVIS16_val': TestDAVIS('../DB/DAVIS', '2016', 'val')
    }

    for key, dataset in datasets.items():
        evaluator = evaluation.Evaluator(dataset)
        evaluator.evaluate(model, os.path.join('outputs', key))


def test_fbms(model):
    test_set = TestFBMS('../DB/FBMS' + '/TestSet')
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1)
    model.cuda()
    ious = []

    for vos_data in test_loader:
        imgs = vos_data['imgs'].cuda()
        flows = vos_data['flows'].cuda()
        masks = vos_data['masks']
        path = vos_data['path'][0]
        seq = path.split('/')[-1]
        valid_frames = vos_data['valid_frames']
        os.makedirs('outputs/FBMS_test/{}'.format(seq), exist_ok=True)

        # get iou of each sequence
        with torch.no_grad():
            vos_out = model(imgs[:, valid_frames], flows[:, valid_frames])
        iou = 0
        count = 0
        for i in range(0, masks.size(1)):
            if torch.sum(masks[0, i]) == 0:
                continue
            tv.utils.save_image(vos_out['masks'][0, i].float().cpu(), 'outputs/FBMS_test/{}/{}_{:05d}.png'.format(seq, seq, int(valid_frames[i])))
            iou = iou + torch.sum(masks[0, i] * vos_out['masks'][0, i].cpu()) / torch.sum((masks[0, i] + vos_out['masks'][0, i].cpu()).clamp(0, 1))
            count = count + 1
        print('{} iou: {:.5f}'.format(seq, iou / count))
        ious.append(iou / count)

    # calculate overall iou
    print('total seqs\' iou: {:.5f}\n'.format(sum(ious) / len(ious)))


def test_ytobj(model):
    test_set = TestYTOBJ('../DB/YTOBJ')
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1)
    model.cuda()
    ious = {'aeroplane': [], 'bird': [], 'boat': [], 'car': [], 'cat': [], 'cow': [], 'dog': [], 'horse': [], 'motorbike': [], 'train': []}
    total_iou = 0
    total_count = 0

    for vos_data in test_loader:
        imgs = vos_data['imgs'].cuda()
        flows = vos_data['flows'].cuda()
        masks = vos_data['masks']
        path = vos_data['path'][0]
        cls = path.split('/')[-2]
        seq = path.split('/')[-1]
        valid_frames = vos_data['valid_frames']
        os.makedirs('outputs/YTOBJ/{}/{}'.format(cls, seq), exist_ok=True)

        # get iou of each sequence
        with torch.no_grad():
            vos_out = model(imgs[:, valid_frames], flows[:, valid_frames])
        iou = 0
        count = 0
        for i in range(0, masks.size(1)):
            if torch.sum(masks[0, i]) == 0:
                continue
            tv.utils.save_image(vos_out['masks'][0, i].float().cpu(), 'outputs/YTOBJ/{}/{}/{:04d}.png'.format(cls, seq, int(valid_frames[i]) + 1))
            iou = iou + torch.sum(masks[0, i] * vos_out['masks'][0, i].cpu()) / torch.sum((masks[0, i] + vos_out['masks'][0, i].cpu()).clamp(0, 1))
            count = count + 1
        if count == 0:
            continue
        print('{}_{} iou: {:.5f}'.format(cls, seq, iou / count))
        ious[cls].append(iou / count)
        total_iou = total_iou + iou / count
        total_count = total_count + 1

    # calculate overall iou
    for cls in ious.keys():
        print('class: {} seqs\' iou: {:.5f}'.format(cls, sum(ious[cls]) / len(ious[cls])))
    print('total seqs\' iou: {:.5f}'.format(total_iou / total_count))


def main():
    #########################
    # for reproducibility
    #########################
    seed = 19971007
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # define model
    model = TMO().eval()
    model = torch.nn.DataParallel(model)

    # training stage
    if options.train:
        train_duts_davis(model)

    # testing stage
    if options.test:
        model.load_state_dict(torch.load('trained_model/duts_davis_best.pth', map_location='cuda:0'))
        with torch.no_grad():
            test_davis(model)
            test_fbms(model)
            test_ytobj(model)


if __name__ == '__main__':
    main()
