import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple


graph = tf.get_default_graph()


def smile_detecting(face, model):
    """ Detecting presence or lack of smile on the face and returning apopriate string """
    neural_network_input_size = 48
    face = face[:, :, 0]
    mean_image = np.mean(face)
    std_image = np.std(face)
    face = (face - mean_image) / std_image
    face = face.reshape(1, neural_network_input_size, neural_network_input_size, 1)
    with graph.as_default():
        prediction = model.predict(face)
    return str(prediction[0, 0])


def is_gray(img: np.ndarray) -> bool:
    """ Check if given image is in grayscale color space. """
    return len(img.shape) == 2


def to_gray(img: np.ndarray) -> np.ndarray:
    """ Convert image to the grayscale color space. """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def crop(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    """ Crop out input image's fragment specified by the rectangle. """
    x, y, w, h = rect
    return img[x:x + w, y:y + h]


def is_rgb(image):
    """ Check if given image is in rgb color space. """
    rgb_dimension = 3
    return len(image.shape) == rgb_dimension


def to_rgb(image):
    """ Convert image to the rgb color space. """
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def extract_faces(img, cascade):
    """ Extracting faces and returning them in a list"""
    faces = cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    faces_list = []
    for (x, y, w, h) in faces:
        faces_list.append(img[x:x + w, y:y + h])
    return faces_list


def face_as_net_input(face, image_size):
    """ Converting given face to input valid to the model"""
    face_input = convert_to_colorspace([face], "grayscale")[0]
    face_input = cv2.resize(face_input, image_size)
    face_input = np.expand_dims(face_input, -1)
    return face_input


def convert_to_colorspace(images, color_space="rgb"):
    """
    Convert list of input images to the target colorspace
    :param images: list of images to convert
    :param color_space: target colorspace
    :return: converted images
    """
    # Convert to rgb space if requested.
    if color_space == "rgb":
        new_images = [
            image if is_rgb(image) else to_rgb(image) for image in images
        ]
        return new_images
    # Convert to grayscale otherwise.
    new_images = [
            image if is_gray(image) else to_gray(image)
            for image in images
        ]
    return new_images
