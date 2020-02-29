# 3D Reconstruction from 2D Views

This is an open source package for generating 3D voxelized objaect from 2D images. 
  
The slides of this project can be found at the following link [Insight Artificial Intelligence Project  MVP](http://bit.ly/2dto3Dpresentation).

### Project Demo
This project takes in any 3-Channel RGB image and converts it into 3D-voxelated objects.
![3-channel RGB Image](https://i.ibb.co/CMN03Ld/pixels.jpg=50x50)3-channel RGB image
![3D Voxels](https://i.ibb.co/921YCJs/Voxels-svg.png=50x50)3D voxels
![Demo Link](https://i.ibb.co/gJzSvzB/ezgif-com-video-to-gif-4.gif)

## Datasets

We use the [ShapeNet](https://www.shapenet.org/) and [Pix3D](http://pix3d.csail.mit.edu/) in our experiments, which are available below:

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
git clone https://github.com/roysidhartha/Pix2Vox.git
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
