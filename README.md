# Pix2Vox

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/hzxie/Pix2Vox.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/hzxie/Pix2Vox/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/hzxie/Pix2Vox.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/hzxie/Pix2Vox/alerts/)

This repository contains the source code for the paper [Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images](https://arxiv.org/abs/1901.11153). The follow-up work [Pix2Vox++: Multi-scale Context-aware 3D Object Reconstruction from Single and Multiple Images](https://arxiv.org/abs/2006.12250) has been published in *International Journal of Computer Vision (IJCV)*.

![Overview](https://infinitescript.com/wordpress/wp-content/uploads/2019/04/Pix2Vox-Overview.jpg)

## Cite this work

```
@inproceedings{xie2019pix2vox,
  title={Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images},
  author={Xie, Haozhe and 
          Yao, Hongxun and 
          Sun, Xiaoshuai and 
          Zhou, Shangchen and 
          Zhang, Shengping},
  booktitle={ICCV},
  year={2019}
}
```

## Datasets

We use the [ShapeNet](https://www.shapenet.org/) and [Pix3D](http://pix3d.csail.mit.edu/) datasets in our experiments, which are available below:

- ShapeNet rendering images: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
- ShapeNet voxelized models: http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
- Pix3D images & voxelized models: http://pix3d.csail.mit.edu/data/pix3d.zip

## Pretrained Models

The pretrained models on ShapeNet are available as follows:

- [Pix2Vox-A](https://gateway.infinitescript.com/?fileName=Pix2Vox-A-ShapeNet.pth) (457.0 MB)
- [Pix2Vox-F](https://gateway.infinitescript.com/?fileName=Pix2Vox-F-ShapeNet.pth) (29.8 MB)

## Prerequisites

#### Clone the Code Repository

```
git clone https://github.com/hzxie/Pix2Vox.git
```

#### Install Python Denpendencies

```
cd Pix2Vox
pip install -r requirements.txt
```

#### Update Settings in `config.py`

You need to update the file path of the datasets:

```
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/path/to/Datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH            = '/path/to/Datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'
__C.DATASETS.PASCAL3D.ANNOTATION_PATH       = '/path/to/Datasets/PASCAL3D/Annotations/%s_imagenet/%s.mat'
__C.DATASETS.PASCAL3D.RENDERING_PATH        = '/path/to/Datasets/PASCAL3D/Images/%s_imagenet/%s.JPEG'
__C.DATASETS.PASCAL3D.VOXEL_PATH            = '/path/to/Datasets/PASCAL3D/CAD/%s/%02d.binvox'
__C.DATASETS.PIX3D.ANNOTATION_PATH          = '/path/to/Datasets/Pix3D/pix3d.json'
__C.DATASETS.PIX3D.RENDERING_PATH           = '/path/to/Datasets/Pix3D/img/%s/%s.%s'
__C.DATASETS.PIX3D.VOXEL_PATH               = '/path/to/Datasets/Pix3D/model/%s/%s/%s.binvox'
```

## Get Started

To train Pix2Vox, you can simply use the following command:

```
python3 runner.py
```

To test Pix2Vox, you can use the following command:

```
python3 runner.py --test --weights=/path/to/pretrained/model.pth
```

If you want to train/test Pix2Vox-F, you need to checkout to `Pix2Vox-F` branch first.

```
git checkout -b Pix2Vox-F origin/Pix2Vox-F
```

## License

This project is open sourced under MIT license.
