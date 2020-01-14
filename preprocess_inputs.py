import cv2
import numpy as np

#note: check the pre-trained model shape input, e.g / shape: [1x3x256x456] = [BxCxHxW]
#B - batch size, C - number of channels, H - image heightm W - image width
#https://docs.openvinotoolkit.org/latest/_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html

def preprocessingimage(input_image, height, width):
    image = cv2.resize(input_image, (width,height))
    image = image.tranpose(2,0,1)  #2,0,1 is shuffle dimension to convert h*w*c to c*h*w. it brings the 2nd index to 0th index then 0th to 1st one and 1st to 2nd one.
    image = image.reshape(1,3,height,width)
    return image

def pose_estimation(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related pose estimation model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the pose estimation model
    preprocessed_image = preprocessingimage(preprocessed_image,256,456)

    return preprocessed_image


def text_detection(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related text detection model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the text detection model
    preprocessed_image = preprocessingimage(preprocessed_image,768,1280)

    return preprocessed_image


def car_meta(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related car metadata model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the car metadata model
    preprocessed_image = preprocessingimage(preprocessed_image,72,7)
    return preprocessed_image
