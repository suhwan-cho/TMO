# TMO

This is the official PyTorch implementation of our paper:

> **Treating Motion as Option to Reduce Motion Dependency in Unsupervised Video Object Segmentation**, *WACV 2023*\
> Suhwan Cho, Minhyeok Lee, Seunghoon Lee, Chaewon Park, Donghyeong Kim, Sangyoun Lee\
> Link: [[WACV]](https://openaccess.thecvf.com/content/WACV2023/papers/Cho_Treating_Motion_as_Option_To_Reduce_Motion_Dependency_in_Unsupervised_WACV_2023_paper.pdf) [[arXiv]](https://arxiv.org/pdf/2209.03138.pdf)

> **Treating Motion as Option with Output Selection for Unsupervised Video Object Segmentation**, *arXiv 2023*\
> Suhwan Cho, Minhyeok Lee, Jungho Lee, MyeongAh Cho, Sangyoun Lee\
> Link: [[arXiv]](https://arxiv.org/pdf/2309.14786.pdf)

<img src="https://github.com/user-attachments/assets/1482cb7b-f1d4-4c70-8f98-cb1e6f093c38" width=800>

You can also find other related papers at [awesome-video-object-segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Abstract
In unsupervised VOS, most state-of-the-art methods leverage motion cues obtained from optical flow maps in addition to appearance cues. However, as they are overly dependent on
motion cues, which may be unreliable in some cases, they cannot achieve stable prediction. To overcome this limitation, we design a novel **motion-as-option network** that is not
much dependent on motion cues and a **collaborative network learning strategy** to fully leverage its unique property. Additionally, an **adaptive output selection algorithm** is
proposed to maximize the efficacy of the motion-as-option network at test time. 


## Preparation
1\. Download 
[DUTS](http://saliencydetection.net/duts/#org3aad434), 
[DAVIS](https://davischallenge.org/davis2017/code.html),
[FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets),
[YouTube-Objects](https://data.vision.ee.ethz.ch/cvl/youtube-objects),
and [Long-Videos](https://www.kaggle.com/datasets/gvclsu/long-videos)
from the official websites.

2\. Estimate and save optical flow maps from the videos using [RAFT](https://github.com/princeton-vl/RAFT).

3\. For convenience, I also provide the pre-processed
[DUTS](https://drive.google.com/file/d/1Q-bvC1XM0cAp41a1oTSwhRsy8o6titr7/view?usp=drive_link), 
[DAVIS](https://drive.google.com/file/d/1kx-Cs5qQU99dszJQJOGKNb-wD_090q6c/view?usp=drive_link), 
[FBMS](https://drive.google.com/file/d/1Zgt5ouwFeTpMTemfNeEFz7uEUo77e2ml/view?usp=drive_link), 
[YouTube-Objects](https://drive.google.com/file/d/1t_eeHXJ30TWBNmMzE7vfS0izEafiBfgn/view?usp=drive_link),
and [Long-Videos](https://drive.google.com/file/d/1gZm1QBT_6JmHhphNrxuSztcqkm_eI6Sq/view?usp=drive_link).

4\. Replace dataset paths in "run.py" file with your dataset paths.


## Training
1\. Open the "run.py" file.

2\. Specify the model version.

3\. Verify the training settings.

4\. Start **TMO** training!
```
python run.py --train
```


## Testing
1\. Open the "run.py" file.

2\. Specify the model version and "aos" option.

3\. Choose a pre-trained model.

4\. Start **TMO** testing!
```
python run.py --test
```


## Attachments
[pre-trained model (rn101)](https://drive.google.com/file/d/1GzdzVndz_J9RPnJLZFtRNHqYQD_ecs6r/view?usp=drive_link)\
[pre-trained model (mitb1)](https://drive.google.com/file/d/1tftAPgpiQ4L4IdvaVcdif0XJJSjqVjLs/view?usp=drive_link)\
[pre-computed results](https://drive.google.com/file/d/1MDcykrDGkxbBf82tNxq7Rc6ERP3kvNEl/view?usp=drive_link)


## Note
Code and models are only available for non-commercial research purposes.\
If you have any questions, please feel free to contact me :)
```
E-mail: suhwanx@gmail.com
```