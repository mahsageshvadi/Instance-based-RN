import os
import sys
import cv2
import importlib
import numpy as np
import math
import mrcnn.model as modellib
import importlib
import sklearn

from sklearn.metrics import mean_squared_error
from mrcnn.config import Config


import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from keras.optimizers import SGD, Adam

from  openpyxl import Workbook
import time, argparse
from utils.non_local import non_local_block
import pickle

ROOT_DIR = os.path.abspath(".")

image_height = 100
image_width = 100
#####################################


parser = argparse.ArgumentParser()

parser.add_argument("--name", default= 'Color_bar')          
                                                         
a = parser.parse_args()

Experiment_name = a.name


if Experiment_name == 'Base_bar':
    thickness = np.random.randint(1, 3)
    channel = 1
    color = False
    max_num_obj = 6
    color_value = 0
    #ICRN_model
    #ICRNModelPath

if Experiment_name == 'Number_bar_E2':
    thickness = np.random.randint(1, 3)
    channel = 1
    color = False
    max_num_obj = 12
    color_value = 0
    #ICRN_model
    #ICRNModelPath

if Experiment_name == 'Color_bar_E2':
    thickness = -1
    channel = 3
    color = True
    max_num_obj = 6
    color_value = np.random.uniform(0.0, 0.9,size = (max_num_obj,3))
    #ICRN_model
    #ICRNModelPath


if Experiment_name == 'Number_bar':
    thickness = 1
    channel = 1
    color = False
    max_num_obj = 9
    color_value = 0
    #ICRN_model
    #ICRNModelPath

elif Experiment_name == 'Color_bar':
    thickness = -1
    channel = 3
    color = True
    max_num_obj = 6
    color_value = np.random.uniform(0.0, 0.9,size = (max_num_obj,3))
    np.random.shuffle(color_value)
    #ICRN_model
    #ICRNModelPath

elif Experiment_name == 'Stroke_width_bar':
    thickness_test  = [2, 3]
    thickness=thickness_test[np.random.randint(len(thickness_test))]
    channel = 1
    color = False
    max_num_obj = 6
    color_value = 0
    #ICRN_model
    #ICRNModelPath





model = "ICRN"

image_num = 200
min_num_obj = 3
############################################

#datasetGenerator = importlib.import_module(dir + '.Dataset_generator')
#ICRN_config_dir = importlib.import_module(dir + '.ICRNConfigure')
#ICRN_config = ICRN_config_dir.Config()
#ICRN_model  = importlib.import_module(dir + '.Net_IRNm').Build_IRN_m_Network()


if color == True:
    #RCNNMODEL_DIR = os.path.join(ROOT_DIR, "/raid/mpsych/cqa/MaskRCNNWeights/bars/mask_rcnn_color.h5")
    RCNNMODEL_DIR = os.path.join(ROOT_DIR, "MaskRCNNlogs/BarChart_color")
else:
   # RCNNMODEL_DIR = os.path.join(ROOT_DIR, "/raid/mpsych/cqa/MaskRCNNWeights/bars/mask_rcnn_gray.h5")
    RCNNMODEL_DIR = os.path.join(ROOT_DIR, "MaskRCNNlogs/BarChart_color/logs_bargrayscale12")


# Load Mask RCNN 


class ShapesConfig(Config):

    NAME = "CQA"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    NUM_CLASSES = 1 + 1  

    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  

    TRAIN_ROIS_PER_IMAGE = 32

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 5
    
maskrcnnconfig = ShapesConfig()
maskrcnnconfig.display()


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

maskrcnn_model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=RCNNMODEL_DIR)

maskrcnn_model_path = maskrcnn_model.find_last()

# Load trained weights
maskrcnn_model.load_weights(maskrcnn_model_path, by_name=True)

##############################################################
#The model 

def Level1_Module():
    input = Input(shape=(image_height, image_width, channel))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = non_local_block(x)   # non local block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = non_local_block(x)   # non local block
    return Model(inputs=input, outputs=x)

# Level2 module is to compute the ratio of a pair.
# Level2 has one NON-LOCAL block.
def Level2_Module(w,h,c):
    print("Level2:", w,h,c)

    inputA = Input(shape=(w, h, c))
    inputB = Input(shape=(w, h, c))

    combined = keras.layers.concatenate([inputA, inputB])   # concatenate them.
    z = Conv2D(64, (3, 3), activation='relu',padding='same')(combined)
    z = Conv2D(64, (3, 3), activation='relu', padding='same')(z)
    z = non_local_block(z)   # non local block
    #
    z = Flatten()(z)
    z = Dense(256, activation="relu")(z)
    z = Dropout(0.5)(z)
    z = Dense(1, activation="linear")(z)  # output the ratio of this pair.

    return Model(inputs=[inputA, inputB], outputs=z)



# IRN_m is final network to estimate the ratio vectors from multiple input instances.
def Build_IRN_m_Network():
    # input layers.
    input_layers = []
    # the first 'obj_num' inputs are corresponding to the input sub-charts.
    for i in range(max_num_obj):
        input = Input(shape=(image_height, image_width, channel), name="input_{}".format(i))
        input_layers.append(input)

    # The last input layer is used for representing R1=(o1/o1)=1.0 which is just a constant.
    # Here, I would use an extra input layer which is 1-dim and always equal to 1.0 rather than directly using a contant.
    # It makes same effect and can avoid some strange compile errors. (I only use TensorFlow before, not way familiar to Keras.)
    R1_one_input = Input(shape=(1,),name="input_constant_scalar1",dtype='float32')   # always equal to 1.0.
    input_layers.append(R1_one_input)

    # First extract individual features.
    individual_features = []
    level1 = Level1_Module()  # build a level1 module
    for i in range(max_num_obj):
        x = level1(input_layers[i])
        individual_features.append(x)

    # Use a Level2 module to predict pairwise ratios.
    level2 = Level2_Module(w=int(individual_features[0].shape[1]),
                           h=int(individual_features[0].shape[2]),
                           c=int(individual_features[0].shape[3]))

    ratio_p_layers = [R1_one_input]   # pairwise ratio vector. put in' R1=(o1/o1)=1.0 '.
    for i in range(max_num_obj-1): # compute the ratio of each neighbor pair.
        x = level2(inputs = [individual_features[i], individual_features[i+1]])
        ratio_p_layers.append(x)

    print("ratio_p_layers", len(ratio_p_layers), ratio_p_layers[-1].shape)

    # Compute the ratios relative to the first object by using MULTIPLY() operation.
    ratio_layers = [R1_one_input]  # put in R1=1.0.
    i = 1
    while i<len(ratio_p_layers):
        x = keras.layers.Multiply()(ratio_p_layers[:i+1])   # R1*R2*...Ri
        i+=1
        ratio_layers.append(x)

    # divide the maxinum of 'ratio_layers' to get the final results.
    max = keras.layers.maximum(ratio_layers)
    z = keras.layers.concatenate(ratio_layers)
    z = keras.layers.Lambda(lambda x: x[0]/x[1])([z, max])

    print("output layer: ", z.shape)

    return Model(inputs=input_layers, outputs=z)



















##############################################################

# generate images


_images = np.ones((max_num_obj, image_num, image_height, image_width, channel), dtype='float32')
_labels = []
number_of_bars = []


def get_segmented_image(segments_bbs, image):
    
    removed_image = image.copy()
    for i in range(len(segments_bbs)):
        toremovesegment = segments_bbs[i]
    
        x1 = toremovesegment[0]
        y1  = toremovesegment[2]
        x2 = toremovesegment[1]
        y2 = toremovesegment[3]

        removed_image[x1 : y1, x2 : y2] = (1, 1, 1)
        removed_image[x1 : y1, x2 : y2] =removed_image[x1 : y1, x2 : y2] +  np.random.uniform(0, 0.05, (abs(y1-x1), abs(y2-x2),3))
        _min = 0.0  # because the image is not 0/1 black-and-white image, is a RGB image.
        _max = removed_image.max()
        removed_image -= _min
        removed_image[x1 : y1, x2 : y2] /= (_max - _min)
   
    return removed_image


def get_segmented_image_grayscale(segments_bbs, image):
    
    padding = 2
    removed_image = image.copy()
    for i in range(len(segments_bbs)):
        toremovesegment = segments_bbs[i]
    
        x1 = toremovesegment[0]
        y1  = toremovesegment[2]
 
        x2 = toremovesegment[1]
        y2 = toremovesegment[3]

        y1 = 100
        removed_image[0 : y1, x2-padding : y2+padding ] = (1, 1, 1)
        removed_image[0 : y1, x2-padding : y2+padding ] =removed_image[0 : y1, x2-padding : y2+padding ] +  np.random.uniform(0, 0.05, (abs(y1), abs(y2-x2 + padding + padding),3))
        _min = 0.0  
        _max = removed_image.max()
        removed_image -= _min
        removed_image[0 : y1, x2-padding : y2+padding ] /= (_max - _min)
   
    removed_image_gray_scale = removed_image[:,:,1:2]

    return removed_image_gray_scale



def GenerateOneBarChart(num, size = image_width):

    image = np.ones(shape=(size, size, channel))
    subImages = [np.ones(shape=(size,size,channel)) for i in range(max_num_obj)]
    heights = np.random.randint(10,80,size=(num))

    barWidth = int( (size-5*(num+1)-4)//num * (np.random.randint(60,100)/100.0) )
    barWidth = max(barWidth, 6)
    spaceWidth = (size-(barWidth)*num)//(num+1)

    sx = (size - barWidth*num - spaceWidth*(num-1))//2
    for i in range(num):

        sy = size - 1
        ex = sx + barWidth
        ey = sy - heights[i]

        if color is True:
                cv2.rectangle(image,(sx,sy),(ex,ey),color_value[i],thickness)
                cv2.rectangle(subImages[i],(sx,sy),(ex,ey),color_value[i],thickness)
        else:
            cv2.rectangle(image,(sx,sy),(ex,ey),color_value,thickness)
            cv2.rectangle(subImages[i],(sx,sy),(ex,ey),color_value,thickness)
        sx = ex + spaceWidth

    # add noise
    noises = np.random.uniform(0, 0.05, (size, size,channel))
    image = image + noises
    _min = image.min()
    _max = image.max()
    image -= _min
    image /= (_max - _min)

    for i in range(len(subImages)):
        noises = np.random.uniform(0, 0.05, (size, size, channel))
        subImages[i] = subImages[i] + noises
        _min = subImages[i].min() if i<num else 0.0
        _max = subImages[i].max()
        subImages[i] -= _min
        subImages[i] /= (_max - _min)
    #
    heights = heights.astype('float32')
    max_height = max(heights)

    for i in range(len(heights)):
        heights[i] /= max_height
    return image, subImages, heights



for i in range(image_num):

        print('******************')
        print(i)
        image, subimages, featureVector = GenerateOneBarChart(
                num=np.random.randint(min_num_obj, max_num_obj + 1))
        
        featureVector = np.array(featureVector)
        
        color_img = np.zeros((image.shape[0], image.shape[1], 3))

        if color is False:
            color_img[:,:,0] = image[:,:,0]
            color_img[:,:,1] = image[:,:,0]
            color_img[:,:,2] = image[:,:,0]
            results = maskrcnn_model.detect([color_img], verbose=1)

        else:
            results = maskrcnn_model.detect([image], verbose=1)

           # image = color_img

        r = results[0]
        arr = r['rois']
        segments_bbs = arr[arr[:,1].argsort()]

        segments = []
        for p in range(len(r['rois'])):
            if color is True:
                segments.append(get_segmented_image([x for x in segments_bbs if ((x != segments_bbs[p]).any())], image))
            else:
                segments.append(get_segmented_image_grayscale([x for x in segments_bbs if ((x != segments_bbs[p]).any())], color_img))

            
        subImages = [np.ones(shape=(image_width,image_width,channel)) for i in range(max_num_obj)]
        for count in range(len(r['rois'])):

            if count< max_num_obj:
                subImages[count] = segments[count]
            
        for t in range(max_num_obj):
            _images[t][i] = subimages[t]

        number_of_bars.append(len(r['rois']))
        
        label = np.zeros(max_num_obj, dtype='float32')
        label[:len(featureVector)] = featureVector
        _labels.append(label)
        
_labels = np.array(_labels, dtype='float32')


if model == "ICRN":

    x_test = _images
    x_test -= .5

    x_test = [x_test[i] for i in range(max_num_obj)]
    x_test.append(np.ones(image_num))
    input_test = [x_test[i] for i in range(max_num_obj)]
    input_test.append(np.ones(input_test[0].shape[0]))

ICRNModelPath = os.path.join(ROOT_DIR, "results/ICRN/" + Experiment_name)

ICRN_model = Build_IRN_m_Network()
m_optimizer = Adam(0.0001)
ICRN_model.compile(loss='mse', optimizer=m_optimizer)

ICRN_model.load_weights(ICRNModelPath + '/model.h5')





predict_Y = ICRN_model.predict(x=input_test, batch_size=1)
y = _labels

MLAE = np.log2(sklearn.metrics.mean_absolute_error( predict_Y * 100, y * 100) + .125)

print(MLAE)

predictFile = open(ROOT_DIR + "/MLAE_results/_predicted_results_{}.txt".format(Experiment_name),'w')
predictFile.write(str(MLAE) + '\t')
predictFile.close()

# Image generation 

# Segment data based on mask 

# get test data 


# load the ICRN model 
# load RN model 
# load VGG model 
# load other model 


# predict 

# save results 





