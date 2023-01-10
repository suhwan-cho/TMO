# TMO

This is the official PyTorch implementation of our paper:

> **Treating Motion as Option to Reduce Motion Dependency in Unsupervised Video Object Segmentation**, *WACV'23*\
> [Suhwan Cho](https://github.com/suhwan-cho), [Minhyeok Lee](https://github.com/Hydragon516), [Seunghoon Lee](https://github.com/iseunghoon), [Chaewon Park](https://github.com/codnjsqkr), [Donghyeong Kim](https://github.com/donghyung87), Sangyoun Lee

URL: [[Official]](https://openaccess.thecvf.com/content/WACV2023/html/Cho_Treating_Motion_as_Option_To_Reduce_Motion_Dependency_in_Unsupervised_WACV_2023_paper.html) [[arXiv]](https://arxiv.org/abs/2209.03138)\
PDF: [[Official]](https://openaccess.thecvf.com/content/WACV2023/papers/Cho_Treating_Motion_as_Option_To_Reduce_Motion_Dependency_in_Unsupervised_WACV_2023_paper.pdf) [[arXiv]](https://arxiv.org/pdf/2209.03138.pdf)

<img src="https://user-images.githubusercontent.com/54178929/208474605-7586894f-11cf-4e38-ac21-75a78216c22d.png" width=800>

```
@article{TMO,
  title={Treating Motion as Option to Reduce Motion Dependency in Unsupervised Video Object Segmentation},
  author={Cho, Suhwan and Lee, Minhyeok and Lee, Seunghoon and Park, Chaewon and Kim, Donghyeong and Lee, Sangyoun},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5140--5149},
  year={2023}
}
```
You can also find other related papers at [awesome-video-object-segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Abstract
In unsupervised VOS, most state-of-the-art methods leverage motion cues obtained from optical flow maps in addition to appearance cues. However, as they are overly dependent on motion cues, which may be unreliable in some cases, they cannot achieve stable prediction. To overcome this limitation, we design a novel network that operates regardless of motion availability, termed as a **motion-as-option network**. Additionally, to fully exploit the property of the proposed network that motion is not always required, we introduce a **collaborative network learning strategy**. As motion is treated as option, fine and accurate segmentation masks can be consistently generated even when the quality of the flow maps is low.

## Preparation
1\. Download [DUTS](http://saliencydetection.net/duts/#org3aad434) for network training.

2\. Download [DAVIS](https://davischallenge.org/davis2017/code.html) for network training and testing.

3\. Download [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/) for network testing.

4\. Download [YouTube-Objects](https://data.vision.ee.ethz.ch/cvl/youtube-objects/) for network testing.

5\. Save optical flow maps from DAVIS, FBMS, and YouTube-Objects using [RAFT](https://github.com/princeton-vl/RAFT).

6\. For convenience, I also provide the pre-processed [DUTS](https://drive.google.com/file/d/1ygndCXvqAXHV26NeJdw6Fub_TFIKIx5C/view?usp=share_link), [DAVIS](https://drive.google.com/file/d/1WReuSYQ7pORUbxda18-Rka076OX9mPdx/view?usp=sharing), [FBMS](https://drive.google.com/file/d/1_SAzXEuPDv9tPIdFdZD-ZXU_BgebAmDs/view?usp=sharing), and [YouTube-Objects](https://drive.google.com/file/d/1fwW3vxRQ-uOg_qzzoYql6fGVzMvBcqlY/view?usp=sharing).

7\. Replace dataset paths in *"run.py"* file with your dataset paths.


## Training
1\. Select training datasets in *"run.py"* file.

2\. Run **TMO** training!!
```
python run.py --train
```


## Testing
1\. Make sure the [pre-trained model](https://drive.google.com/file/d/12k0iZhcP6Z8RdGKCKHvlZq5g9kNtj8wA/view?usp=share_link) is in your *"trained_model"* folder.

2\. Run **TMO** testing!!
```
python run.py --test
```

3\. (Optional) Download [pre-computed results](https://drive.google.com/file/d/1bWrxXiE5_0Kz-i63xoRk68r8cJL8kMgY/view?usp=sharing).



## Note
Code and models are only available for non-commercial research purposes.

If you have any questions, please feel free to contact me :)
```
E-mail: chosuhwan@yonsei.ac.kr
```
