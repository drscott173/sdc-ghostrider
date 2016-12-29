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
signal_scale = 1
train_split = 0.95
test_image = "./test_train/test/IMG/center_2016_12_26_10_59_29_903.jpg"
gid = "00b4903a97d4b5e0859e13d01d014b8f2dd1ebc9fbf9188650a15efe983f8596term"

##
## Image augmentation
##

def yuv2hls(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2HLS)
    return img2

def hls2yuv(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
    img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2YUV)
    return img1

def yuv2hsv(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    return img1

def hsv2yuv(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2YUV)
    return img1

def add_random_shadow(image):
    rows,cols,_ = image.shape
    top_y = cols*np.random.uniform()
    top_x = 0
    bot_x = rows
    bot_y = cols*np.random.uniform()
    image_hls = yuv2hls(image)
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
    image = hls2yuv(image_hls)   
    return image

def augment_brightness_camera_images(image):
    # expects input image as YUV
    image1 = yuv2hsv(image)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image2 = hsv2yuv(image1)
    return image2

def trans_image(img,steer,trans_range):
    # Translation
    image = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    rows, cols, _ = image.shape
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    img_final = cv2.cvtColor(image_tr, cv2.COLOR_BGR2YUV)
    return img_final,steer_ang

def augment_image(X, y):
    #print(X.shape)
    camera = np.random.randint(3)
    if (camera == 0): 
        # left 
        X = X[0]
        y += 0.25
    if (camera == 1):
        # center
        X = X[1]
    if (camera == 2):
        # right
        X = X[2]
        y += -0.25
    #print("Chosen X shape", X.shape)
    img = X
    if (np.random.randint(4) == 0):
        img_shadows = add_random_shadow(img)
        img_bright = augment_brightness_camera_images(img_shadows)
        img_tr, y = trans_image(img_bright, y, 50)
        img = img_tr
        if (np.random.randint(2) == 0):
            img = cv2.flip(img_tr,1)
            y = -y
    #img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return img, y

def write_aug(X,y,fname):
    x1 = cv2.cvtColor(X, cv2.COLOR_YUV2BGR)
    x2 = cv2.resize(x1, (128, 128), interpolation=cv2.INTER_AREA)
    cv2.putText(x2, "y={}".format(y), (2,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
    cv2.imwrite(fname, x2) 

def test_augmentation(X, y):
    write_aug(X[0],y,"./tmp/00augment.png")
    for i in range(0,64):
        x1, y1 = augment_image(X,y)
        write_aug(x1,y1,"./tmp/augment{}.png".format(i+1))

def region_of_interest(img):
    #
    # We crop and resize to 66x200 per NVidia, assuming a 160x320 input
    #
    img_roi = img[55:135,40:280]
    img = cv2.resize(img_roi,(200,66))
    return img

def process(img):
    # Here's the basic image pipeline for a single frame
    #
    # We clip the center of the image and conert to three channels of YUV
    #
    out = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    out1 = region_of_interest(out)
    return out1

def image_to_data(subdir, path):
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
    path = train_dir+subdir+"/"+(path.strip())
    data = process(cv2.imread(path).astype('uint8'))
    return data

def epoch_size():
    size = 0
    for d in [x[len(train_dir):] for x in glob(train_dir+"*")]:
        base = train_dir+d+"/"
        csv = base+"driving_log.csv"
        df=pd.read_csv(csv, sep=',',header=None, names=csv_keys)
        size += len(df)
    return size

def gen_data(subdir):
	# 
	# Using the directory structure above, we generate all the "data.p" file
	# that will store the processed images and sensor readings ready for training.
	# 'subdir' refers to a training session that we give names like left for
	# veering off left, normal for normal driving, right for veering right.
	#
	# We store [X,y] as a pickled file data.p if it doesn't already exist.
	#
    base = train_dir+subdir+"/"
    csv = base+"driving_log.csv"
    data_path = base+"data.p"
    chunk_size = 1000
    paths = glob(base+"data*.p")
    if not os.path.exists (data_path):
        paths = []
        df=pd.read_csv(csv, sep=',',header=None, names=csv_keys)
        X_raw = df[['left','center','right']]
        y_raw = df[['angle']]
        n_samples = len(X_raw)
        for chunk in range(0,n_samples,chunk_size):
            n = int(chunk/chunk_size)
            if n > 0:
                data_path = "{}data{}.p".format(base, n)
            paths.append(data_path)
            lo = n*chunk_size
            hi = min(n_samples, lo+chunk_size)
            X_df = X_raw.ix[lo:hi,:].apply(lambda row: row.apply(lambda path: image_to_data(subdir, path)),
                                    axis=1)
            X_df = X_df[0:chunk_size]
            y_df = y_raw[lo:hi]
            print("Sizes ",len(X_df),len(y_df))
            X = np.array(X_df.values)
            y = np.array(y_df.values)[:,0]
            print("Saving X shape ", X.shape, "y shape", y.shape, "for", subdir)
            pickle.dump([X,y], open(data_path, "wb"))
            print("Created ",data_path)
    return paths

def gen_all_data(verbose=True):
	# Loop through all training data and make sure we have data.p files
	# of prepared image data X and steering angle y [X,y].
    for d in [x[len(train_dir):] for x in glob(train_dir+"*")]:
        if verbose: 
            print("Generating data for "+d)
        gen_data(d)

# Reshape data into tensors for training

def get_input_shape():
    first_dir = [x[len(train_dir):] for x in glob(train_dir+"*")][0]
    base = train_dir+first_dir+"/"
    with open(base+"data.p", mode='rb') as f:
        X_d,y_d = pickle.load(f)
        X_d1 = X_d[:,1] # center camera
        X_d, y_d = reshape_xy([X_d1[0]], [y_d[0]])
        return X_d.shape[1:]

def Xy_generator(validate=False):
    #
    # Generate all data resident in files within the train/ directory
    # using a train/test split where "train_split" is the percentage
    # of the data we want to use for training.  
    #
    # If we set "validate" to true, we yield the validation data.
    # If "validate'" is false, we yield the training data.
    #
    ishape = get_input_shape()
    chosen = np.zeros((1, ishape[0], ishape[1], ishape[2]))
    while 1:
        for d in [x[len(train_dir):] for x in glob(train_dir+"*")]:   #iterate over trainin sets
            for data_path in gen_data(d):
                with open(data_path, mode='rb') as f:    #iterate over chunks
                    print("\nVisiting ",data_path)
                    X_d,y_d = pickle.load(f)
                    X_train, X_test, y_train, y_test = train_test_split(X_d, y_d, 
                        test_size=(1.0-train_split), random_state=42)
                    #print("Num train", len(X_train), "Num test", len(X_test))
                    if validate:
                        X_d = X_test
                        y_d = y_test
                    else:
                        X_d = X_train
                        y_d = y_train
                    # images are left, center, right.  tweak angle by +/- 0.07 for
                    # left and right
                    for n in range(0,len(X_d)):
                        chosen[0,:,:,:], y_i = augment_image(X_d[n], y_d[n])
                        yield (chosen, y_i)

def batch_generator(batch_size=32, validate=False, skip_pr=0.0):
    ishape = get_input_shape()
    X_batch = np.zeros((batch_size, ishape[0], ishape[1], ishape[2]))
    #X_batch = np.zeros((batch_size, 64, 64, 3))
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

def squarify(X):
    n = X.shape[0]
    out = np.zeros((n,64,64,3))
    for i in range(0,X.shape[0]):
        img = X[i,:,:,:]
        out[i,:,:,:] = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return out

def reshape_xy(X,y):
    #
    # X is a single-dimension array of images, where each image is a 3-plane WxH image
    # stored as a [w,h,3] array.  y is a single dimension array of label values.
    #
    # We return a tensor X [n, w, h, 3] and a tensor y [n].
    #
    image_shape = X[0].shape
    n_samples = len(X)
    #X1 = np.zeros((n_samples, image_shape[2], image_shape[0], image_shape[1]))
    X1 = np.zeros((n_samples, image_shape[0], image_shape[1], image_shape[2]))
    y1 = np.zeros((n_samples))
    for i in range(0,n_samples):
        #for j in range(0,2):
        #    X1[i,j,:,:] = np.array(X[i])[:,:,j]
        X1[i,:,:,:] = np.array(X[i])
        y1[i] = signal_scale*y[i]
    return X1, y1


def comma_model(shape):
    #
    # Build the self-driving CNN model from comma.ai,
    # about cut in half.
    #
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=shape, 
        name='Normalization'))
    model.add(Convolution2D(3, 1, 1, name='Color'))
    model.add(Convolution2D(16, 5, 5, subsample=(3,3), name='Conv1', activation='elu'))
    model.add(Convolution2D(32, 3, 3, subsample=(2,2), name='Conv2', activation='elu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2,2), name='Conv3'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Activation('elu'))
    model.add(Dense(512, name='FC1'))
    model.add(Dropout(0.5))
    model.add(Activation('elu'))
    model.add(Dense(1, name='output'))
    model.summary()
    return model

def build_model(shape):
    return nvidia_model(shape)

def vivek_model(shape):
    #
    # Build the self-driving CNN model from the NVidia paper
    #
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(64,64,3), name='Normalization'))

    model.add(Convolution2D(3, 1, 1, name='Conv1'))

    model.add(Convolution2D(32, 3, 3, activation="elu", name="Conv2"))
    model.add(Convolution2D(32, 3, 3, activation="elu", name="Conv3"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, activation="elu", name="Conv4"))
    model.add(Convolution2D(64, 3, 3, activation="elu", name="Conv5"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, activation="elu", name="Conv6"))
    model.add(Convolution2D(128, 3, 3, activation="elu", name="Conv7"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512, name='FC1'))
    model.add(Dense(64, name='FC2'))
    model.add(Dense(16, name='FC3'))

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
    #cv2.imshow('image',out)
    #cv2.waitKey()
    ##img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    X, _ = reshape_xy([out], [0])
    p = model.predict(X)
    return p[0][0]/signal_scale

