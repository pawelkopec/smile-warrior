import base64
from flask import Flask, request, render_template
from flask_cors import cross_origin
from image_utils import smile_detecting, face_as_net_input, extract_faces
from keras.models import load_model
import tensorflow as tf
import argparse
import numpy as np
import cv2

# creates a Flask application, named app
app = Flask(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Input of model, weights and cascade path')
    parser.add_argument('--cascade', required=True, type=str,
                        help='OpenCV cascade file path')
    parser.add_argument('--model', required=True, type=str,
                        help='Model address path', default="model.hdf5")
    parser.add_argument('--weight', required=True, type=str,
                        help='Model weights path')
    return parser.parse_args()


# a route where we will display a welcome message via an HTML template
@app.route("/")
def hello():
    return render_template('main.html')


def prepare_faces(image):
    """ Prepare image to the requirements of the model"""
    facecascade = cv2.CascadeClassifier(cascade)
    faces = extract_faces(image, facecascade)
    if len(faces) == 0:
        raise ValueError("-1")
    data_in = face_as_net_input(faces[0], tuple([48, 48])).astype('f')
    return data_in


def data_uri_to_cv2_img(url):
    """ Convert url data to opencv format """
    image_data = url.split(',')[1]
    decoded = base64.b64decode(image_data)
    nparr = np.fromstring(decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img


@app.route('/classify', methods=['POST'])
@cross_origin()
def classify():
    print("got request")
    data_url = request.form.get('imgBase64')
    img_cv2 = data_uri_to_cv2_img(data_url)
    try:
        img_ndarray_faces = prepare_faces(img_cv2)
    except ValueError as v:
        return str(v)
    r = smile_detecting(img_ndarray_faces, model)
    return r


# run the application
if __name__ == "__main__":
    args = parse_args()
    cascade = args.cascade
    model = args.model
    weight = args.weight
    graph = tf.get_default_graph()
    model = load_model(model)
    model.load_weights(weight)
    print('loaded everything :)')
    app.run(debug=False)
