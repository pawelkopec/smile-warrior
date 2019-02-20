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


cascpath = "haarcascade_frontalface_default.xml"
modelpath = "model.hdf5"
weightspath = "weights.03.hdf5"


graph = tf.get_default_graph()
model = load_model('model.hdf5')
model.load_weights('weights.03.hdf5')
print('loaded everything :)')


def parse_args():
    parser = argparse.ArgumentParser(description='Input of model, weights and cascade path')
    parser.add_argument('--cascade-path', '-c', required=False, type=str,
                        help='OpenCV cascade file path, default set to: ' +
                             cascpath, default=cascpath)
    parser.add_argument('--model', '-t', required=False, type=str,
                        help='Model address address, default' +
                        'set to: ' + modelpath,
                        default=modelpath)
    parser.add_argument('--weight', '-t', required=False, type=str,
                        help='Model weights address address, default' +
                        'set to: ' + weightspath,
                        default=weightspath)
    return parser.parse_args()


# a route where we will display a welcome message via an HTML template
@app.route("/")
def hello():
    return render_template('main.html')


def prepare_faces(image):
    """ Prepare image to the requirements of the model"""
    facecascade = cv2.CascadeClassifier(cascpath)
    faces = extract_faces(image, facecascade)
    if len(faces) == 0:
        raise ValueError("Couldn't find face on an image.")
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
    cascpath = args.cascade_path
    model = args.modelpath
    weight = args.weightspath
    app.run(debug=False)
