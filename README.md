# DASNet: Dual attentive fully convolutional siamese networks for change detection of high-resolution satellite images


<!-- Pytorch implementation of Change Detection as described in [DASNet: Dual attentive fully convolutional siamese networks for change detection of high-resolution satellite images](https://arxiv.org/pdf/2003.03608.pdf).-->
The architecture:
<img src="DASNET/img/p1.jpg" width="600px" hight="400px" />

## Requirements

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
We give an example of the directory structure in the .example and the values of the label images need to be 0 and 1.


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

## Citation
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
