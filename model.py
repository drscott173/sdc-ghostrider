import numpy as np
import cv2
import math
import os
import matplotlib.image as mpimg
import pandas as pd
import pickle
import os.path
import json
from glob import glob
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

# Some Hyper Parameters
csv_keys = ['center','left','right','angle','throttle','brake','speed']
train_dir = 'train/'
batch_size = 32  # Batch size when training
nb_epoch = 20 # Number of epochs for training
train_split = 0.95
test_image = "./test_train/test/IMG/center_2016_12_26_10_59_29_903.jpg"
use64x64 = 0 # Set to 1 to use Vivek's innovative 64x64 model for images

##
## Image augmentation
##

def bgr2hls(img):
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    return img2

def hls2bgr(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
    return img1

def bgr2hsv(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img1

def hsv2bgr(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img1

def add_random_shadow(image):
    #
    # Add a random shadow to a BGR image to pretend
    # we've got clouds or other interference on the road.
    #
    rows,cols,_ = image.shape
    top_y = cols*np.random.uniform()
    top_x = 0
    bot_x = rows
    bot_y = cols*np.random.uniform()
    image_hls = bgr2hls(image)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y)-(bot_x - top_x)*(Y_m-top_y) >=0)] = 1
    random_bright = .25+.7*np.random.uniform()
    if (np.random.randint(2) ==1 ):
        random_bright = .5
        cond1 = (shadow_mask==1)
        cond0 = (shadow_mask==0)
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright 
    image = hls2bgr(image_hls)   
    return image

def augment_brightness_camera_images(image):
    #
    # expects input image as BGR, adjusts brightness to 
    # pretend we're in different lighting conditions.
    #
    image1 = bgr2hsv(image)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image2 = hsv2bgr(image1)
    return image2

def trans_image(img,steer,trans_range):
    # 
    # Shift image up or down a bit within trans_range pixels,
    # filling missing area with black.  IMG is in BGR format.
    #
    rows, cols, _ = img.shape
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    img_tr = cv2.warpAffine(img,Trans_M,(cols,rows))
    return img_tr, steer_ang

def region_of_interest(img):
    #
    # We crop and resize to 66x200 per NVidia, assuming a 160x320 input
    #
    img_roi = img[55:135,40:280]
    img = cv2.resize(img_roi,(200,66))
    return img

def image_to_data(path):
    #
    # Our simulation log data records a 'path' of an image, which we modified
    # by making the path relative to a parent directory.  Here we pass in that
    # directory name and append it to our train directory.
    #
    # train_dir
    #    subdir
    #       IMG/  - images for subdir
    #       driving_log.csv - log of image name in IMG and sensor readings
    #    subdir
    #       IMG/  - images for subdir
    #       driving_log.csv - log of image name in IMG and sensor readings
    #    ... etc
    #
    # This routine converts a subdir and path into an array for training.
    #
    #path = train_dir+subdir+"/"+(path.strip())
    data = process(cv2.imread(path).astype('uint8'))
    return data

def augment_image(imageList, y):
    # 
    # We "augment" images during training.
    #
    # imageList is a list of camera image pathnames [left, center, right]
    # y is the label for these three images
    #
    # We choose one of the camera images, augment it to simulate
    # potential environments, then return a chosen image and label y.
    #
    camera = np.random.randint(3)
    if (camera == 0): 
        # left 
        X = imageList[0]
        y += 0.25
    if (camera == 1):
        # center
        X = imageList[1]
    if (camera == 2):
        # right
        X = imageList[2]
        y += -0.25
    #print("Chosen X shape", X.shape)
    img_data = cv2.imread(X).astype('uint8')
    img = augment_brightness_camera_images(img_data)
    if (np.random.randint(4) == 0):
        img_shadows = add_random_shadow(img)
        img_tr, y = trans_image(img_shadows, y, 50)
        img = img_tr
        if (np.random.randint(2) == 0):
            img = cv2.flip(img_tr,1)
            y = -y
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_clipped = region_of_interest(img_yuv)
    if use64x64:
        img64 = cv2.resize(img_clipped, (64, 64), interpolation=cv2.INTER_AREA)
        return img64, y
    return img_clipped, y

def process(img):
    #
    # We "process" images during replay, without augmentation.
    #
    # Here's the basic image pipeline for a single frame
    #
    # We clip the center of the image and convert to three channels of YUV,
    # then resize to 64x64 if needed.
    #
    out = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    out1 = region_of_interest(out)
    if use64x64:
        return cv2.resize(out1, (64, 64), interpolation=cv2.INTER_AREA)
    return out1

#
# A couple routines to debug image augmentation
#

def write_aug(X,y,fname):
    # 
    # Write out a 128x128 thumbnail of the augmented image in X with label y to fname
    #
    x1 = cv2.cvtColor(X, cv2.COLOR_YUV2BGR)
    x2 = cv2.resize(x1, (128, 128), interpolation=cv2.INTER_AREA)
    cv2.putText(x2, "y={}".format(y), (2,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
    cv2.imwrite(fname, x2) 

def test_augmentation(X, y):
    #
    # Run through 64 augmentations of X,y and dump the png images
    # into the tmp directory, starting with original as 00augment
    # and then augment0-augment63 for all others.
    #
    write_aug(X[0],y,"./tmp/00augment.png")
    for i in range(0,64):
        x1, y1 = augment_image(X,y)
        write_aug(x1,y1,"./tmp/augment{}.png".format(i+1))

def epoch_size():
    #
    # Add up all the driving logs and see how many images we have,
    # then return this count as the size of one training epoch.
    #
    size = 0
    for d in [x[len(train_dir):] for x in glob(train_dir+"*")]:
        base = train_dir+d+"/"
        csv = base+"driving_log.csv"
        df=pd.read_csv(csv, sep=',',header=None, names=csv_keys)
        size += len(df)
    return size

# Reshape data into tensors for training

def get_input_shape():
    #
    # Look at the first image we've recorded, process it,
    # and return the image shape we'll feed into the network.
    #
    first_dir = glob(train_dir+"*")[0][len(train_dir):]
    image_dir = train_dir+first_dir+"/IMG/"
    first_image = glob(image_dir+'*')[0]
    raw = cv2.imread(first_image).astype('uint8')
    return process(raw).shape

def paths_and_steering_from_subdir(d):
    #
    # Return an array of [left, center, right] paths for images,
    # along with a corresponding array of [angle] steering angles
    # taken from subdirectory "d" of the train_dir training directory.
    #
    # We look for driving_log.csv from the simulator.  This code
    # uses the base pathname in the driving log, then searches for
    # this image in train_dir/d/IMG/* for images.
    # 
    base = train_dir+d+"/"
    csv = base+"driving_log.csv"
    df=pd.read_csv(csv, sep=',',header=None, names=csv_keys)
    paths_raw = df[['left','center','right']]
    paths_raw = paths_raw.apply(
        lambda row: 
            row.apply(
                lambda f: base+"IMG/"+os.path.basename(f)))
    paths_raw = paths_raw.values
    steering_raw = df[['angle']].values
    return paths_raw, steering_raw

def Xy_generator(validate=False):
    #
    # Generate all data resident in files within the train/ directory
    # using a train/test split where "train_split" is the percentage
    # of the data we want to use for training.  
    #
    # If we set "validate" to true, we yield the validation data.
    # If "validate'" is false, we yield the training data.
    #
    # Each input sample has three image pathnames for X: [p1, p2, p3] and 
    # on floating point value for the steering label y: [steer]
    #
    #
    ishape = get_input_shape()
    chosen = np.zeros((1, ishape[0], ishape[1], ishape[2]))
    dirs = [x[len(train_dir):] for x in glob(train_dir+"*")]
    while 1:
        np.random.shuffle(dirs)
        for d in dirs:   #iterate over trainin sets
            print("Visiting {}".format(d))
            X_d, y_d = paths_and_steering_from_subdir(d)
            X_train, X_test, y_train, y_test = train_test_split(X_d, y_d, 
                test_size=(1.0-train_split), random_state=42)
            if validate:
                X_d = X_test
                y_d = y_test
            else:
                X_d = X_train
                y_d = y_train
            for n in range(0,len(X_d)):
                chosen[0,:,:,:], y_i = augment_image(X_d[n], y_d[n])
                yield (chosen, y_i)

def batch_generator(batch_size=32, validate=False, skip_pr=0.0):
    #
    # Rotate through all the generated single test cases of Xy_generator,
    # chunking them into a single "batch" tensor for smoother fitting
    # and faster processing.
    #
    ishape = get_input_shape()
    X_batch = np.zeros((batch_size, ishape[0], ishape[1], ishape[2]))
    y_batch = np.zeros((batch_size))
    ptr = 0
    xy = Xy_generator(validate)
    while 1:

        skip_this = 1
        while skip_this:
            X_batch[ptr], y_batch[ptr] = next(xy)
            if abs(y_batch[ptr]) < 0.1:
                if np.random.uniform() > skip_pr:
                    skip_this = 0
            else:
                skip_this = 0

        ptr += 1
        if (ptr % batch_size) == 0:
            yield (X_batch, y_batch)
            ptr = 0

def reshape_xy(X,y):
    #
    # X is a single-dimension array of images, where each image is a 3-plane WxH image
    # stored as a [w,h,3] array.  y is a single dimension array of label values.
    #
    # We return a tensor X [n, w, h, 3] and a tensor y [n].
    #
    image_shape = X[0].shape
    n_samples = len(X)
    X1 = np.zeros((n_samples, image_shape[0], image_shape[1], image_shape[2]))
    y1 = np.zeros((n_samples))
    for i in range(0,n_samples):
        X1[i,:,:,:] = np.array(X[i])
        y1[i] = y[i]
    return X1, y1

def build_model(shape):
    #
    # Add a colorspace transformation before the NVidia model,
    # then add pooling and dropouts in the pipeline in lieu of
    # successive downsampling with convolutions.
    #
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=shape, name='Normalization'))
    model.add(Convolution2D(3, 1, 1, name='ColorSpace'))

    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='elu', name='Conv1'))
    model.add(Convolution2D(36, 5, 5, activation='elu', name='Conv2'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(48, 5, 5, activation='elu', name='Conv3'))
    model.add(Convolution2D(64, 3, 3, activation='elu', name='Conv4'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, activation='elu', name='Conv5'))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(100, name='FC1'))
    model.add(Dense(50, name='FC2'))
    model.add(Dense(10, name='FC3'))
    model.add(Dense(1, name='output'))
    model.summary()
    return model

def nvidia_model(shape):
    #
    # Build the self-driving CNN model from the NVidia paper
    #
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=shape, name='Normalization'))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='elu', name='Conv1'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='elu', name='Conv2'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='elu', name='Conv3'))
    model.add(Convolution2D(64, 3, 3, activation='elu', name='Conv4'))
    model.add(Convolution2D(64, 3, 3, activation='elu', name='Conv5'))
    model.add(Flatten())
    model.add(Dense(100, name='FC1'))
    model.add(Dense(50, name='FC2'))
    model.add(Dense(10, name='FC3'))
    model.add(Dense(1, name='output'))
    model.summary()
    return model

def save_model(model):
    #
    # Save the model weights in .h5 and the layout in .json per instructions.
    #
    with open("./model.json", "w") as f:
        f.write(model.to_json())
        #son.dump(model.to_json(), f)
    model.save_weights("./model.h5", True)
    return model

def train_model(model, skip_pr = 0.8):
    # 
    # Train our model on images X and driving angles y.
    # Scramble and choose a train/test split of 80/20.
    #
    samples = epoch_size()
    train_epoch = int(samples*train_split)
    test_epoch = samples - train_epoch
    epochs = 0
    while epochs < nb_epoch:
        model.fit_generator(
            batch_generator(skip_pr=skip_pr),
            samples_per_epoch = train_epoch,
            validation_data = batch_generator(validate=True, skip_pr=skip_pr),
            nb_val_samples = test_epoch,
            max_q_size = 3,
            verbose=1,
            nb_epoch=1)
        save_model(model)
        skip_pr *= 0.5
        epochs += 1
    return model

def learn(model=None, skip_pr=1.0):
    #
    # Use this routine to train and save a new model.
    #
    if model is None:
        model = build_model(get_input_shape())
        model.compile(loss='mse', optimizer=Adam(), metrics=['accuracy'])
    return save_model(train_model(model, skip_pr=skip_pr))

def load_model():
    #
    # Likewise, load a model here.
    #
    with open('./model.json', mode='r') as f:
        loaded_json = f.read()
    model = model_from_json(loaded_json)
    model.compile(loss='mse', optimizer=Adam(), metrics=['accuracy'])

    # load weights into new model
    model.load_weights("./model.h5")
    return model

def predict_steering(model, img):
    # 
    # Given a model and a raw input image, return the predicted steering angle.
    #
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #cv2.imshow('image',img_bgr)
    #cv2.waitKey()
    out = process(img_bgr)
    X, y = reshape_xy([out], [0])
    #cv2.imshow('image',out)
    #cv2.waitKey()
    return model.predict(X)[0][0]

