import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
    
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "CQA"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = ShapesConfig()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("CQA", 1, "bar")


        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bars= self.GenerateBarData(width, height)
            self.add_image("CQA", image_id=i, path=None,
                           width=width, height=height, bars=bars)#,
                          # bg_color=bg_color, shapes=shapes)
                
    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bars = info['bars']
        image = np.ones(shape=(info['height'], info['width'], 3))
        image= self.drawImage(image, bars, info['height'],  info['width'])
        return image/255
    
    
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        bars = info['bars']
        count = len(bars)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (bar_name, _, dims) in enumerate(info['bars']):
 
            mask[:, :, i:i+1] = self.drawImage(mask[:, :, i:i+1].copy(),
                                                [bars[i]], info['height'], info['width'], True)

        
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index('bar') for s in bars])
        return mask.astype(bool), class_ids.astype(np.int32)
        
    
    def GenerateBarData(self, height, width):
    
        min_num_obj = 3
        max_num_obj = 6
        num=np.random.randint(min_num_obj, max_num_obj + 1)
        #todo: change max_obj_num for more bars
        max_obj_num = 6
        colors = np.random.uniform(0.0, 0.9,size = (max_obj_num,3))
        heights = np.random.randint(10,80,size=(num))

        barWidth = int( (width-3*(num+1)-3)//num * (np.random.randint(50,100)/100.0) )
        barWidth = max(barWidth, 4)
        spaceWidth = (width-(barWidth)*num)//(num+1)

        sx = (width - barWidth*num - spaceWidth*(num-1))//2
        bars = []

        for i in range(num):

            sy = width - 1
            ex = sx + barWidth
            ey = sy - heights[i]

            bar_name = 'bar_{}'.format(i)
            bars.append((bar_name, colors[i], (sx, sy, ex, ey)))
            sx = ex + spaceWidth

        return bars
     
    
    def drawImage(self, image, bars, height, width, mask=False):
        
        for bar in bars:
            sx, sy, ex, ey = bar[2]
            if mask== False:
                color = bar[1]
            else:
                color = 1
            cv2.rectangle(image,(sx,sy),(ex,ey),color,-1)
        if mask is False:
            channel  = 3
            noises = np.random.uniform(0, 0.05, (height, width,channel))
            image = image + noises
            _min = 0.0
            _max = image.max()
            image -= _min
            image /= (_max - _min)
        
        return image * 255
        
dataset_train = ShapesDataset()
#<<<<<<< HEAD
dataset_train.load_shapes(10000, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

dataset_val = ShapesDataset()
dataset_val.load_shapes(100, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
#=======
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
#>>>>>>> 3210294c4dc0f1a5098b38b6a00c193fdfcf5d3b
dataset_val.prepare()


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
    
#<<<<<<< HEAD
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'     
    
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=30, 
            layers='heads')

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=60, 
            layers="all")


    
    
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')

#>>>>>>> 3210294c4dc0f1a5098b38b6a00c193fdfcf5d3b
model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
model.keras_model.save_weights(model_path)






