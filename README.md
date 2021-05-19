# DASNet: Dual attentive fully convolutional siamese networks for change detection of high-resolution satellite images
Change detection is a basic task of remote sensing image processing. The research objective is to identify the change information of interest and filter out the irrelevant change information as interference factors. Recently, the rise in deep learning has provided new tools for change detection, which have yielded impressive results. However, the available methods focus mainly on the difference information between multitemporal remote sensing images and lack robustness to pseudo-change information. To overcome the lack of resistance in current methods to pseudo-changes, in this paper, we propose a new method, namely, dual attentive fully convolutional Siamese networks (DASNet), for change detection in high-resolution images. Through the dual attention mechanism, long-range dependencies are captured to obtain more discriminant feature representations to enhance the recognition performance of the model. Moreover, the imbalanced sample is a serious problem in change detection, i.e., unchanged samples are much more abundant than changed samples, which is one of the main reasons for pseudo-changes. We propose the weighted double-margin contrastive loss to address this problem by punishing attention to unchanged feature pairs and increasing attention to changed feature pairs. The experimental results of our method on the change detection dataset (CDD) and the building change detection dataset (BCDD) demonstrate that compared with other baseline methods, the proposed method realizes maximum improvements of 2.9% and 4.2%, respectively, in the F1 score.


You can visit the paper via https://ieeexplore.ieee.org/document/9259045/ or arxiv @ https://arxiv.org/abs/2003.03608

<!-- Pytorch implementation of Change Detection as described in [DASNet: Dual attentive fully convolutional siamese networks for change detection of high-resolution satellite images](https://arxiv.org/pdf/2003.03608.pdf).-->
The architecture:


<img src="img/p1.jpg" width="600px" hight="400px" />

## Requirements

Most of problems in the issue list are caused by the version of python or pytoch.
We have updated the source code to fit new version of pytorch.
Hope our repo is useful to you.

- Python3.6
- Pytorch 1.3.1 (see: [pytorch installation instuctions](http://pytorch.org/))
- torchvision 0.4.2

## Datasets
This repo is built for remote sensing change detection. We report the performance on two datasets.

- CDD
 - paper: [Change detection in remote sensing images using conditional adversarial networks](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-2/565/2018/isprs-archives-XLII-2-565-2018.pdf)
 
- BCDD
 - paper: [ Fully Convolutional Networks for Multisource Building Extraction From an Open Aerial and Satellite Imagery Data Set](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8444434)

 
### Directory Structure
 
File Structure is as follows:

```
$T0_image_path/*.jpg
$T1_image_path/*.jpg
$ground_truth_path/*.jpg
```
##################


NOTE: We give an example of the directory structure in the .example and the values of the label images need to be 0 and 1.
IF you did not revise it, our model will lost it's mind.


##################

## Pretrained Model
The backbone model and pretrained models for CDD and BCDD can be download from [[googledriver]](https://drive.google.com/open?id=1iTsmLDCWcNm6odchkpmZY6dSq7dEpQBP) [[baidudisk]](https://pan.baidu.com/s/1GFkBXvVKgD1IqLYYeioX_w )   password:86of


## Training
```shell
cd $CD_ROOT
python train.py
```
## Testing
```shell
cd $CD_ROOT
python test.py
```

## Citation
If our repo is useful to you, please cite our published paper as follow:
```
Bibtex
@article{chen2020dasnet,
    title={DASNet: Dual attentive fully convolutional siamese networks for change detection of high resolution satellite images},
    author={Chen, Jie and Yuan, Ziyang and Peng, Jian and Chen, Li and Huang, Haozhe and Zhu, Jiawei and Lin, Tao and Li, Haifeng},
    journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
    DOI = {10.1109/JSTARS.2020.3037893},
    year={2020},
    type = {Journal Article}
}

Endnote
%0 Journal Article
%A Chen, Jie
%A Yuan, Ziyang
%A Peng, Jian
%A Chen, Li
%A Huang, Haozhe
%A Zhu, Jiawei
%A Lin, Tao
%A Li, Haifeng
%D 2020
%T DASNet: Dual attentive fully convolutional siamese networks for change detection of high resolution satellite images
%B IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
%R 10.1109/JSTARS.2020.3037893
%! DASNet: Dual attentive fully convolutional siamese networks for change detection of high resolution satellite images
```
