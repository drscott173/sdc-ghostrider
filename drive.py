import argparse
import base64
import json
from model import predict_steering

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import cv2

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = float(data["steering_angle"])
    throttle = float(data["throttle"])
    speed = float(data["speed"])

    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    #image.show()
    image_array = np.array(image)  # critical!  np.asarray() doesn't work on Mac

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    new_steer = predict_steering(model, image_array)

    # Steering is designed for a speed of 8.  Let's reduce the angle if we're going faster
    if speed > 8:
        theta = new_steer*(8.0/speed)
    else:
        theta = new_steer

    # Try to stay at 10mph
    throttle = max(0.0, min(0.5, (10.0-speed)/5.0))

    #steering_angle = 0
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    
    print(theta, throttle)
    send_control(theta, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', 
        type=str,
        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    #app = socketio.Middleware(sio, app)
    #eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())
        # was model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
