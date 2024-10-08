# TMO

This is the official PyTorch implementation of our paper:

> **Treating Motion as Option to Reduce Motion Dependency in Unsupervised Video Object Segmentation**, *WACV 2023*\
> Suhwan Cho, Minhyeok Lee, Seunghoon Lee, Chaewon Park, Donghyeong Kim, Sangyoun Lee\
> Link: [[WACV]](https://openaccess.thecvf.com/content/WACV2023/papers/Cho_Treating_Motion_as_Option_To_Reduce_Motion_Dependency_in_Unsupervised_WACV_2023_paper.pdf) [[arXiv]](https://arxiv.org/pdf/2209.03138.pdf)

> **Treating Motion as Option with Output Selection for Unsupervised Video Object Segmentation**, *arXiv 2023*\
> Suhwan Cho, Minhyeok Lee, Jungho Lee, MyeongAh Cho, Sangyoun Lee\
> Link: [[arXiv]](https://arxiv.org/pdf/2309.14786.pdf)

<img src="https://user-images.githubusercontent.com/54178929/208474605-7586894f-11cf-4e38-ac21-75a78216c22d.png" width=800>

You can also find other related papers at [awesome-video-object-segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Abstract
In unsupervised VOS, most state-of-the-art methods leverage motion cues obtained from optical flow maps in addition to appearance cues. However, as they are overly dependent on
motion cues, which may be unreliable in some cases, they cannot achieve stable prediction. To overcome this limitation, we design a novel **motion-as-option network** that is not
much dependent on motion cues and a **collaborative network learning strategy** to fully leverage its unique property. Additionally, an **adaptive output selection algorithm** is
proposed to maximize the efficacy of the motion-as-option network at test time. 


## Preparation
1\. Download [DUTS](http://saliencydetection.net/duts/#org3aad434) for network training.

2\. Download [DAVIS](https://davischallenge.org/davis2017/code.html) for network training and testing.

3\. Download [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets) for network testing.

4\. Download [YouTube-Objects](https://data.vision.ee.ethz.ch/cvl/youtube-objects) for network testing.

5\. Download [Long-Videos](https://www.kaggle.com/datasets/gvclsu/long-videos) for network testing.

6\. Save optical flow maps of DAVIS, FBMS, YouTube-Objects, and Long-Videos using [RAFT](https://github.com/princeton-vl/RAFT).

7\. For convenience, I also provide the pre-processed [DUTS](https://drive.google.com/file/d/1Q-bvC1XM0cAp41a1oTSwhRsy8o6titr7/view?usp=sharing), [DAVIS](https://drive.google.com/file/d/1kx-Cs5qQU99dszJQJOGKNb-wD_090q6c/view?usp=sharing), [FBMS](https://drive.google.com/file/d/1Zgt5ouwFeTpMTemfNeEFz7uEUo77e2ml/view?usp=sharing), [YouTube-Objects](https://drive.google.com/file/d/1t_eeHXJ30TWBNmMzE7vfS0izEafiBfgn/view?usp=sharing), and [Long-Videos](https://drive.google.com/file/d/1gZm1QBT_6JmHhphNrxuSztcqkm_eI6Sq/view?usp=sharing).

8\. Replace dataset paths in "run.py" file with your dataset paths.


## Training
1\. Move to "run.py" file.

2\. Define model version (ver): 'rn101' or 'mitb1'

3\. Check training settings.

4\. Run **TMO** training!!
```
python run.py --train
```


## Testing
1\. Move to "run.py" file.

2\. Define model version (ver): 'rn101' or 'mitb1'

3\. Use or not adaptive output selection (aos): True or False

4\. Select a pre-trained model that accords with the defined model version.

5\. Run **TMO** testing!!
```
python run.py --test
```

## Attachments
[pre-trained model (rn101)](https://drive.google.com/file/d/1GzdzVndz_J9RPnJLZFtRNHqYQD_ecs6r/view?usp=sharing)\
[pre-trained model (mitb1)](https://drive.google.com/file/d/1tftAPgpiQ4L4IdvaVcdif0XJJSjqVjLs/view?usp=sharing)\
[pre-computed results](https://drive.google.com/file/d/1MDcykrDGkxbBf82tNxq7Rc6ERP3kvNEl/view?usp=sharing)


## Note
Code and models are only available for non-commercial research purposes.\
If you have any questions, please feel free to contact me :)
```
E-mail: suhwanx@gmail.com
```
