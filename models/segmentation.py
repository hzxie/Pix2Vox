import torch.hub as hb
import torch
import numpy as np
import cv2
import glob
import os
from PIL import Image


class Segmentation:
    def __init__(self, seg_cfg):
        self.model_type = seg_cfg['type']
        self.base_network = seg_cfg['base network']
        self.folder_name = seg_cfg['folder name']
        self.save_folder = seg_cfg['save folder']
        self.padding = seg_cfg['padding']
        print(self.folder_name)
        self.category = MASK_CATEGORY_MAPPING[seg_cfg['mask category']]
        self.initialize_variables()
        self.img_RGBA = None

    def run(self):
        self.load_files()

        self.load_model()

        self.configure_transform()

        for file_path in self.file_list:
            self.segment_image(file_path)

        self.write_file_list()

    def segment_image(self, file_path):
        self.load_image(file_path)
        self.input_tensor = self.transform(self.input_image)
        self.input_batch = self.input_tensor.unsqueeze(0)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            self.input_batch = self.input_batch.to('cuda')
            self.model.to('cuda')

        self.predict()
        print("Image category number {}".format(self.category))
        self.get_RGBA_image()
        self.image_count += 1

    def load_model(self):
        self.model = hb.load(self.model_type, self.base_network, pretrained=True)
        self.model.eval()

    def load_files(self):
        self.file_list = glob.glob(os.path.join(os.getcwd(), self.folder_name, "*.*"))
        self.num_images = len(self.file_list)

    def load_image(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        scale_percent = 517 / img.shape[0] * 100 # percent of original size
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

            categories, counts = np.unique(self.mask, return_counts=True)
            print("Categories - {}".format(categories))
            print("Counts - {}".format(counts))
            if len(counts) == 1:
                max_count_index = np.argmax(counts) + 1
            else:
                max_count_index = np.argmax(counts[1:]) + 1

            self.category = categories[max_count_index]

    def configure_transform(self):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_RGBA_image(self):
        image_array = np.array(self.input_image)
        b_channel, g_channel, r_channel = cv2.split(image_array)
        mask_array = np.where(self.mask != self.category, 0, 255)
        mask_array = mask_array.astype(np.uint8)
        self.img_RGBA = cv2.merge((b_channel, g_channel, r_channel, mask_array))
        resized_RGBA = self.add_padding(self.img_RGBA)
        """
        # Extract image width and height for resizing it
        image_width = img_RGBA.shape[1]
        image_height = img_RGBA.shape[0]
        scale_percent = 137 / image_width * 100
        print(scale_percent)
        width = int(img_RGBA.shape[1] * scale_percent / 100)
        height = int(img_RGBA.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_RGBA = cv2.resize(img_RGBA, dim, interpolation=cv2.INTER_AREA)
        """
        img_name = '{:02d}'.format(self.image_count)
        PATH = os.path.join(self.save_folder,img_name+'.png')
        cv2.imwrite(PATH, resized_RGBA)

    def add_padding(self, image):
        desired_size = 137
        extra_padding = self.padding
        im = image
        print("Read image size {}".format(im.shape))

        old_size = im.shape[:2]  # old_size is in (height, width) format
        print("Old Size is {}".format(old_size))

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        print("New Size is {}".format(new_size))

        # new_size should be in (width, height) format

        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2 + extra_padding, delta_h - (delta_h // 2) + extra_padding
        left, right = delta_w // 2 + extra_padding, delta_w - (delta_w // 2) + extra_padding

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        new_im = cv2.resize(new_im, (desired_size, desired_size), interpolation=cv2.INTER_AREA)
        return new_im
        """
        print('final size = {}'.format(new_im.shape))
        plt.imshow(new_im)

        cv2.imwrite('/home/sidroy/Insight/projects/Pix2Vox/datasets/DemoImage/car/car_subfolder/rendering/00.png',
                    new_im)
        """

    def write_file_list(self):
        PATH = os.path.join(self.save_folder,'renderings.txt' )
        with open(PATH, 'w') as filehandle:
            for list_item in range(self.num_images):
                img_name = '{:02d}'.format(list_item)+'.png'
                filehandle.write('%s\n' % img_name)

    def initialize_variables(self):
        self.input_image = None
        self.model = None
        self.transform = None
        self.input_tensor = None
        self.input_batch = None
        self.output = None
        self.output_predictions = None
        self.mask = None
        self.num_images = None
        self.file_list = None
        self.image_count = 0


MASK_CATEGORY_MAPPING = {
    'AEROPLANE': 1, 'BICYCLE': 2, 'BIRD': 3, 'BOAT': 4, 'BOTTLE': 5, 'BUS': 6,'CAR': 7,
    'CAT': 8, 'CHAIR': 9, 'COW': 10, 'TABLE': 11, 'DOG': 12, 'HORSE': 13, 'MOTOR BIKE': 14,
    'PERSON': 15, 'PLANT': 16, 'SHEEP': 17, 'SOFA': 18, 'TRAIN': 19, 'TV': 20,
}
if __name__ == "__main__":
    SEG_CFG = {}
    SEG_CFG['type'] = 'pytorch/vision:v0.5.0'
    SEG_CFG['base network'] = 'deeplabv3_resnet101'
    SEG_CFG['folder name'] = "../load_images"
    SEG_CFG['mask category'] = 'CHAIR'
    SEG_CFG['save folder'] = '../datasets/DemoImage/car/car_subfolder/rendering'
    SEG_CFG['padding'] = 10

    segment = Segmentation(SEG_CFG)
    segment.run()
