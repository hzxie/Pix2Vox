import torch
import numpy as np
import cv2
from PIL import Image

class Segmentation():
    def __init__(self, seg_cfg):
        self.model_type = seg_cfg['type']
        self.base_network = seg_cfg['base network']
        self.filename = seg_cfg['filename']
        self.category = self.category(seg_cfg['mask category'])
        self.initialize_variables()

    def segment_image(self):
        self.load_image()
        self.load_model()
        self.configure_transform()
        self.input_tensor = self.transform(self.input_image)
        self.input_batch = self.input_tensor.unsqueeze(0)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            self.input_batch = self.input_batch.to('cuda')
            self.model.to('cuda')

        self.predict()
        return self.output_predictions

    def load_model(self):
        self.model = torch.hub.load(self.model_type, self.base_network, pretrained=True)
        self.model.eval()

    def load_image(self):
        img = cv2.imread(self.filename, cv2.IMREAD_UNCHANGED)
        scale_percent = 100  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        print("Dimensions of input to segmentation code {}".format(dim))
        # resize image
        self.input_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    def predict(self):
        """
        output is the pytorch tensor output by the segmentation model
        it contains the probability of each pixel being in a certain class
        output predictions is array of pixels with max probability of being in a class
        """
        with torch.no_grad():
            self.output = self.model(self.input_batch)['out'][0]
        self.output_predictions = self.output.argmax(0)
        self.mask = self.output_predictions.cpu()

        print('output shape is {}'.format(self.output.shape))
        print('output_predictions is of shape {}'.format(self.output_predictions.shape))

    def configure_transform(self):
        from torchvision import transforms
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_RGBA_image(self):
        image_array = np.array(self.input_image)
        b_channel, g_channel, r_channel = cv2.split(self.image_array)
        mask_array = np.where(self.mask != 11, 0, 255)
        mask_array = mask_array.astype(np.uint8)

MASK_CATEGORY_MAPPING = {
        'TABLE': 11,
        'CHAIR': 7,
        'CAR': 12
}



    def initialize_variables(self):
        self.input_image = None
        self.model = None
        self.transform = None
        self.input_tensor = None
        self.input_batch = None
        self.output = None
        self.output_predictions = None
        self.mask = None

if __name__ == "__main__":
    SEG_CFG = {}
    SEG_CFG['type'] = 'pytorch/vision:v0.5.0'
    SEG_CFG['base network'] = 'deeplabv3_resnet101'
    SEG_CFG['filename'] = '/home/sidroy/Downloads/table3.jpg'
    SEG_CFG['mask category'] = 'TABLE'

    segment = Segmentation(SEG_CFG)
    segment.segment_image()