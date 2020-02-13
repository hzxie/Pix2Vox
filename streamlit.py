import matplotlib.pyplot as plt
import torch.backends.cudnn
import torch.utils.data
from pprint import pprint
from config import cfg
from core.demo import test_net
import plotly.graph_objects as go
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import requests
import os
import os.path

title = st.title("3D Reconstruction")

@st.cache
def generate_data():
    PATH = 'pretrained_models/Pix2Vox-A-ShapeNet.pth'
    torch.backends.cudnn.benchmark = True
    checkpoint = (torch.load(PATH))
    cfg.CONST.WEIGHTS = './pretrained_models/Pix2Vox-A-ShapeNet.pth'
    generated_volume, rendering_images = test_net(cfg,output_dir='./output')
    volume = generated_volume.reshape(32,32,32)
    volume = volume.squeeze().__ge__(0.5)
    return volume

MODEL_FILENAME = 'pretrained_models/Pix2Vox-A-ShapeNet.pth'
MODEL_DIR = 'pretrained_models'

@st.cache
def download_model_from_web():
    
    #if os.path.isfile(MODEL_FILENAME):
    #    return

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

# let's put sliders to modify view init, each time you move that the script is rerun, but voxels are not regenerated

# TODO : not sure that's the most optimized way to rotate axis but well, demo purpose

azim = st.sidebar.slider("azim", 0, 180, 30, 1)

elev = st.sidebar.slider("elev", 0, 360, 240, 1)

# Download Data
download_model_from_web()

# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
volume = generate_data()
ax.voxels(volume, facecolors='r', edgecolor='k')
ax.view_init(azim, elev)
ax.axis('off')
st.pyplot()


