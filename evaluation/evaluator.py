import utils
import os
import time


class Evaluator(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.img_saver = utils.ImageSaver()
        self.sdm = utils.DAVISLabels()

    def evaluate_video(self, model, video_name, video_parts, output_path):
        for vos_data in video_parts:
            imgs = vos_data['imgs'].cuda()
            flows = vos_data['flows'].cuda()
            files = vos_data['files']

            # inference
            t0 = time.time()
            vos_out = model(imgs, flows)
            t1 = time.time()

            # save output
            for i in range(len(files)):
                fpath = os.path.join(output_path, video_name, files[i])
                data = ((vos_out['masks'][0, i, 0, :, :].cpu().byte().numpy(), fpath), self.sdm)
                self.img_saver.enqueue(data)
        return t1 - t0, imgs.size(1)

    def evaluate(self, model, output_path):
        model.cuda()
        total_seconds, total_frames = 0, 0
        for video_name, video_parts in self.dataset.get_videos():
            os.makedirs(os.path.join(output_path, video_name), exist_ok=True)
            seconds, frames = self.evaluate_video(model, video_name, video_parts, output_path)
            total_seconds = total_seconds + seconds
            total_frames = total_frames + frames
            print('{} done, {:.1f} fps'.format(video_name, frames / seconds))
        print('total fps: {:.1f}\n'.format(total_frames / total_seconds))
        self.img_saver.kill()
