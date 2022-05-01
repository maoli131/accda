# Contains utilities needed to process images/labels from image segmentation datasets

import tensorflow as tf

import numpy as np

import os

import io_utils as io

def process_gta5(img_shape = (1024,512), \
                 img_dir = "./data/GTA5/images/", \
                 label_dir = "./data/GTA5/labels", \
                 target_dir = "./processed-data/1024x512/19_classes/GTA5/train", \
                 target_label_ids = None):
    '''
    
        read_gta5 reads images in the GTA5 data format, and saves these images in
            .npy format to a target directory
        
        img_shape defines the (W,H) format for the images
        
        img_dir is the image containing directory
        label_dir is the label containing directory

        target_dir is the directory where the image and label pairs are going to be stored, reshaped
            to img_shape, and in .npy format

        target_label_ids should represents the final label mapping
    '''
    assert target_label_ids is not None

    img_id = 0
    for img_name in sorted(os.listdir(img_dir)):
        img_path = img_dir + img_name
        labels_path = label_dir + img_name
        
        img, labels = io.read_and_reshape(img_path, labels_path, img_shape, 'GTA5')

        # Every image should be successfully processed
        if img is None:
            continue

        # Standardize labels to label_ids
        labels_final = np.zeros(labels.shape)
        for curr_label in gta5_label_ids:
            ind = labels == gta5_label_ids[curr_label]
            if curr_label in target_label_ids.keys():
                labels_final[ind] = target_label_ids[curr_label]

        # Save the img,labels pair to the target directory
        np.save(target_dir + str(img_id) + "_image.npy", img.astype(np.uint8))
        np.save(target_dir + str(img_id) + "_label.npy", labels_final.astype(np.int8))
        img_id += 1

def process_synthia(img_shape = (1024,512), \
                 img_dir = "./data/SYNTHIA/RGB/", \
                 label_dir = "./data/SYNTHIA/GT/LABELS/", \
                 target_dir = "./processed-data/1024x512/13_classes/SYNTHIA/train/", \
                 target_label_ids = None):
    '''
        read_synthia reads images in the SYNTHIA data format, and saves these images in
            .npy format to a target directory
        
        img_shape defines the (W,H) format for the images
        
        img_dir is the image containing directory
        label_dir is the label containing directory

        target_dir is the directory where the image and label pairs are going to be stored, reshaped
            to img_shape, and in .npy format

        target_label_ids should represents the final label mapping
    '''

    assert target_label_ids is not None
    
    img_id = 0
    for img_name in sorted(os.listdir(label_dir)):
        img_path = img_dir + img_name
        labels_path = label_dir + img_name
        
        img, labels = io.read_and_reshape(img_path, labels_path, img_shape, 'SYNTHIA')

        # Every image should be successfully processed
        if img is None:
            continue

        # Standardize labels to label_ids
        labels_final = np.zeros(labels.shape)
        for curr_label in synthia_label_ids:
            ind = labels == synthia_label_ids[curr_label]
            if curr_label in target_label_ids.keys():
                labels_final[ind] = target_label_ids[curr_label]

        # Save the img,labels pair to the target directory
        np.save(target_dir + str(img_id) + "_image.npy", img.astype(np.uint8))
        np.save(target_dir + str(img_id) + "_label.npy", labels_final.astype(np.int8))
        img_id += 1


def process_cityscapes(img_shape = (1024,512), \
                 img_dir = "./data/CITYSCAPES/images/train/", \
                 label_dir = "./data/CITYSCAPES/labels/train/", \
                 target_dir = "./processed-data/1024x512/13_classes/CITYSCAPES/train/", \
                 target_label_ids = None):
    '''
        read_cityscapes reads images in the CITYSCAPES data format, and saves these images in
            .npy format to a target directory

        In case an image fails loading, the function may return a batch that is smaller than 
        the inputed batch_size.

        img_shape defines the (W,H) format for the images

        img_dir is the image containing directory
        label_dir is the label containing directory

        target_dir is the directory where the image and label pairs are going to be stored, reshaped
            to img_shape, and in .npy format

        target_label_ids should represents the final label mapping
    '''

    assert target_label_ids is not None

    # Gather all training images/labels
    cities = os.listdir(img_dir)

    img_id = 0
    for city in cities:
        curr_img_dir = img_dir + city + "/"
        curr_label_dir = label_dir + city + "/"
        
        for img in os.listdir(curr_img_dir):
            img_path = curr_img_dir + img

            prefix = img.rsplit("_", 1)[0]
            labels_path = curr_label_dir + prefix + "_gtFine_labelIds.png"

            img, labels = io.read_and_reshape(img_path, labels_path, img_shape, 'CITYSCAPES')

            # Every image should be successfully processed
            if img is None:
                continue

            # Standardize labels to label_ids
            labels_final = np.zeros(labels.shape)
            for curr_label in cityscapes_label_ids:
                ind = labels == cityscapes_label_ids[curr_label]
                if curr_label in target_label_ids.keys():
                    labels_final[ind] = target_label_ids[curr_label]

            # Save the img,labels pair to the target directory
            np.save(target_dir + str(img_id) + "_image.npy", img.astype(np.uint8))
            np.save(target_dir + str(img_id) + "_label.npy", labels_final.astype(np.int8))
            img_id += 1


# Final mapping of labels, to be combined between GTA5 and CITYSCAPES
# 19 classes + ignore, similar to https://arxiv.org/pdf/1903.04064.pdf
label_ids_19 = {'ignore': 0,
 'road': 1,
 'sidewalk': 2,
 'building': 3,
 'wall': 4,
 'fence': 5,
 'pole': 6,
 'traffic light': 7,
 'traffic sign': 8,
 'vegetation': 9,
 'terrain': 10,
 'sky': 11,
 'person': 12,
 'rider': 13,
 'car': 14,
 'truck': 15,
 'bus': 16,
 'train': 17,
 'motorcycle': 18,
 'bicycle': 19}

# Final mapping of labels, to be combined between SYNTHIA and CITYSCAPES
# 13 classes + ignore, similar to https://arxiv.org/pdf/1903.04064.pdf
label_ids_13 = {'ignore': 0,
 'road': 1,
 'sidewalk': 2,
 'building': 3,
 'traffic light': 4,
 'traffic sign': 5,
 'vegetation': 6,
 'sky': 7,
 'person': 8,
 'rider': 9,
 'car': 10,
 'bus': 11,
 'motorcycle': 12,
 'bicycle': 13}

# Label mapping in the SYNTHIA dataset
synthia_label_ids = {
    'void'          : 0,
    'sky'           : 1,
    'building'      : 2,
    'road'          : 3,
    'sidewalk'      : 4,
    'fence'         : 5,
    'vegetation'    : 6,
    'pole'          : 7,
    'car'           : 8,
    'traffic sign'  : 9,
    'person'        : 10,   # previously 'pedestrian'
    'bicycle'       : 11,
    'motorcycle'    : 12,
    'parking-slot'  : 13,   # In CITYSCAPES appears as 'parking'. But, in 19 class classification papers, this class is not considered.
    'road-work'     : 14,   # not present in cityscapes
    'traffic light' : 15,
    'terrain'       : 16,
    'rider'         : 17,
    'truck'         : 18,
    'bus'           : 19,
    'train'         : 20,
    'wall'          : 21,
    'lanemarking'   : 22,   # not present in cityscapes
}

# Label mapping in the CITYSCAPES dataset
cityscapes_label_ids = {
    'unlabeled'            :  0 ,
    'ego vehicle'          :  1 ,
    'rectification border' :  2 ,
    'out of roi'           :  3 ,
    'static'               :  4 ,
    'dynamic'              :  5 ,
    'ground'               :  6 ,
    'road'                 :  7 ,
    'sidewalk'             :  8 ,
    'parking'              :  9 ,
    'rail track'           : 10 ,
    'building'             : 11 ,
    'wall'                 : 12 ,
    'fence'                : 13 ,
    'guard rail'           : 14 ,
    'bridge'               : 15 ,
    'tunnel'               : 16 ,
    'pole'                 : 17 ,
    'polegroup'            : 18 ,
    'traffic light'        : 19 ,
    'traffic sign'         : 20 ,
    'vegetation'           : 21 ,
    'terrain'              : 22 ,
    'sky'                  : 23 ,
    'person'               : 24 ,
    'rider'                : 25 ,
    'car'                  : 26 ,
    'truck'                : 27 ,
    'bus'                  : 28 ,
    'caravan'              : 29 ,
    'trailer'              : 30 ,
    'train'                : 31 ,
    'motorcycle'           : 32 ,
    'bicycle'              : 33 ,
    'license plate'        : -1
}

# Label mapping in the GTA5 dataset
gta5_label_ids = cityscapes_label_ids