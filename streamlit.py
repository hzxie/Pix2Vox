import matplotlib.pyplot as plt
#import torch.backends.cudnn
#import torch.utils.data
import torch
from pprint import pprint
from config import cfg
from core.demo import test_net
import models.segmentation as seg
import plotly.graph_objects as go
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import requests
import os
import os.path
import glob
import cv2
from PIL import Image

title = st.title("3D Reconstruction from 2D Views")

description = st.markdown('This application takes 2D imaged views of objects '
                          'and converts it into 3D objects', unsafe_allow_html=False)

description1 = st.markdown('There are two steps involved in this process:')
description2 = st.markdown('*Step 1:* Semantic Image Segmentation using the '
                           '__DeepLab V3+__ architecture trained on COCO dataset')
description3 = st.markdown('*Step 2:* Using __Pix2Vox__ architecture trained on ShapeNet DataSet to convert 2D images'
                           ' to 3D voxel objects', unsafe_allow_html=False)
#@st.cache
def generate_data():
    PATH = 'pretrained_models/Pix2Vox-A-ShapeNet.pth'
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        checkpoint = torch.load(PATH)
    else:
        map_location = torch.device('cpu')
        checkpoint = torch.load(PATH, map_location=map_location)

    cfg.CONST.WEIGHTS = './pretrained_models/Pix2Vox-A-ShapeNet.pth'
    generated_volume, rendering_images = test_net(cfg,output_dir='./output')
    vol = generated_volume.reshape(32,32,32)
    return vol

def config_segmentation(SEG_CFG):
    SEG_CFG['type'] = 'pytorch/vision:v0.5.0'
    SEG_CFG['base network'] = 'deeplabv3_resnet101'
    SEG_CFG['folder name'] = "load_images"
    SEG_CFG['mask category'] = 'CHAIR'
    SEG_CFG['save folder'] = './datasets/DemoImage/car/car_subfolder/rendering'
    SEG_CFG['padding'] = 10

MODEL_FILENAME = 'pretrained_models/Pix2Vox-A-ShapeNet.pth'
MODEL_DIR = 'pretrained_models'

@st.cache
def download_model_from_web():
    
    if os.path.isfile(MODEL_FILENAME):
        return

    try:
        os.mkdir(MODEL_DIR)
    except FileExistsError:
        pass
    
    MODEL_URL = (
        'https://2d-to-3d-pretrained-model.s3-us-west-2.amazonaws.com/Pix2Vox-A-ShapeNet.pth')
    resp = requests.get(MODEL_URL, stream=True)

    with open(MODEL_FILENAME, 'wb') as file_desc:
        for chunk in resp.iter_content(chunk_size=5000000):
            file_desc.write(chunk)

# Download Data
download_model_from_web()

img_file_buffer = st.file_uploader("Upload your image here (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])
print("image file buffer".format(img_file_buffer))

i = 0
if img_file_buffer is not None:
    filelist = glob.glob(os.path.join("load_images", "*.jpg"))
    for f in filelist:
        os.remove(f)
    image = np.array(Image.open(img_file_buffer))

    st.image(image, caption='Uploaded Image',use_column_width = True)
    #cv2.imread(img_file_buffer, cv2.IMREAD_UNCHANGED)
    cv2.imwrite("load_images/"+str(i)+'.jpg', image)
    i += 1
    segment = None
    SEG_CFG = {}
    config_segmentation(SEG_CFG)
    SEG_CFG['padding'] = st.sidebar.selectbox("View Distance", [0, 10, 20, 30, 40])
    # SEG_CFG['folder name'] = img_file_buffer
    segment = seg.Segmentation(SEG_CFG)
    segment.run()
    st.image(segment.img_RGBA, caption='Segmented Image', use_column_width=True)

# let's put sliders to modify view init, each time you move that the script is rerun, but voxels are not regenerated
volume = generate_data()

# TODO : not sure that's the most optimized way to rotate axis but well, demo purpose
description = st.sidebar.markdown('Control Your View Angle Here')
azim = st.sidebar.slider("azimuth angle", -180, 180, -64, 1)
elev = st.sidebar.slider("elevation angle", -180, 180, 111, 1)

# and plot everything
threshold = st.sidebar.selectbox("Threshold: Controls the voxel threshold levels", [0.2, 0.3, 0.4, 0.5, 0.6])
fig = plt.figure()
ax = fig.gca(projection='3d')
volume_plot = volume.squeeze().__ge__(threshold)
ax.voxels(volume_plot, facecolors='r', edgecolor='k')
ax.view_init(azim, elev)
ax.axis('off')
st.pyplot()


