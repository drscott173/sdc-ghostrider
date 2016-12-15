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
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

# Some Hyper Parameters
csv_keys = ['center','left','right','angle','throttle','brake','speed']
train_dir = 'train/'
batch_size = 200  # Batch size when training
nb_epoch = 200 # Number of epochs for training
signal_scale = 1

def process(img):
    # Here's the basic image pipeline for a single frame
    #
    # We convert RGB to grayscale then put on blinders
    # to ignore scenery above and below the driving
    # area.
    #
    xsize = img.shape[1]
    ysize = img.shape[0]
    dx = int(xsize*0.0) # offset from x left, right border
    dy_top = int(ysize*0.35) # offset from y top border
    dy_bottom = int(ysize*0.2) #offset from y bottom 
    center = int(0.5*xsize) 
    vertices = np.array([[(dx,ysize-dy_bottom), 
                          (dx, dy_top), 
                          (xsize, dy_top), 
                          (xsize, ysize-dy_bottom)]], 
                        dtype=np.int32)
    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = region_of_interest(out, vertices)
    out = cv2.resize(out, (0,0), fx=0.5, fy=0.5)
    return out

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color  
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def normalize_image(image):
	#
	# We convert a grayscale image to an array of numbers
	# that vary from [-0.5, 0.5], normalizing grays to a range of 1.0
	# with a median of 0.
	#
    img = np.array(image, dtype=np.float32)
    lo = np.min(img[:,:])*1.0
    hi = np.max(img[:,:])*1.0
    img[:,:] = (img[:,:]-lo)/(hi-lo) - 0.5
    return img

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
    data = normalize_image(process(mpimg.imread(path).astype('uint8')))
    return data

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
    paths = [x[len(base):] for x in glob(base+"data*.p")]
    if not os.path.exists(data_path):
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
    for d in [x[6:] for x in glob("train/*")]:
        if verbose: 
            print("Generating data for "+d)
        with open(gen_data(d), mode='rb') as f:
            X,y = pickle.load(f)
            if verbose:
                print("  X is ", X.shape, " y is ",y.shape)

# Reshape data into tensors for training

def load_all_data():
	#
	# Load all data into memory.  I didn't use generators
	# this time around.
	#
    X = np.array([])
    y = np.array([])
    print("Loading data, please stand by.")
    for d in [x[6:] for x in glob(train_dir+"*")]:
        for data_path in gen_data(d):
            with open(data_path, mode='rb') as f:
                X_d,y_d = pickle.load(f)
                for i in range(1,2):
                    X = np.append(X,X_d[:,i])
                    y = np.append(y,y_d)
    return X, y
    
def reshape_xy(X,y):
	#
	# I stored raw images as a single dimension array, where each entry
	# holds three subarrays.  These subarrays are width x height images.
	# This routine converts an n-elt array [wxh, wxh, ..., wxh] into a tensor of
	# shape [n, w, h, 1].
	#
    image_shape = X[0].shape
    n_samples = len(X)
    X1 = np.zeros((n_samples, image_shape[0], image_shape[1], 1))
    y1 = np.zeros((n_samples))
    for i in range(0,n_samples):
        X1[i,:,:] = X[i].reshape(image_shape[0], image_shape[1], 1)
        y1[i] = signal_scale*y[i]
    return X1, y1

def build_model(shape):
	#
	# Build the self-driving CNN model from the NVidia paper
	#
	model = Sequential()
	model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2,2), input_shape=shape))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2,2)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2,2)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2,2)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2,2)))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Activation('relu'))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

def build_simple_model(shape):
	#
	# We build a simple convolution layer with RELU activtaion,
	# followed by a full-connected layer with 100 controls,
	# then a final output.  KISS.
	#
	model = Sequential()
	model.add(Convolution2D(24, 5, 5, border_mode='same', 
		subsample=(2,2), input_shape=shape))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(1))
	return model

def prefer_turns_in_data(X, y):
	#
	# The NVidia team recommended focusing on turns vs the usual
	# case of "0" angle for going straight.  The more complex models would solve for
	# 0 and struggle with the smaller adjustments in steering.  The
	# accuracy was directly related to the percentage of 0's encountered!
	# Here we ensure that 0's are only half the training set.
	#
	turns = np.where(y != 0)[0]
	straight = np.where(y == 0)[0]
	straight2 = np.random.choice(straight, len(turns))
	sample = np.append(turns,straight2)
	return X[sample], y[sample]

def save_model(model):
	#
	# Save the model weights in .h5 and the layout in .json per instructions.
	#
	with open("./model.json", "w") as f:
		json.dump(model.to_json(), f)
	model.save("./model.h5")
	return model

def train_model(model, X, y):
	# 
	# Train our model on images X and driving angles y.
	# Scramble and choose a train/test split of 80/20.
	#
	epochs = 0
	while epochs < nb_epoch:
		X1, y1 = prefer_turns_in_data(X,y)
		X_train, X_test, y_train, y_test = train_test_split(X1, y1, 
			test_size=0.2, random_state=42)
		X1 = y1 = None
		model.fit(X_train, y_train,
			batch_size=batch_size, nb_epoch=5,
        	verbose=1, validation_data=(X_test, y_test))
		epochs += 5
		save_model(model)

	return model

def learn(model=None):
	#
	# Use this routine to train and save a new model.
	#
	X_raw, y_raw = load_all_data()
	X, y = reshape_xy(X_raw, y_raw)
	if model is None:
		model = build_model(X.shape[1:])
	model.compile(loss='mse', optimizer=Adam())
	return save_model(train_model(model, X, y))

def load_model():
	#
	# Likewise, load a model here.
	#
	with open('./model.json', mode='r') as f:
		model = model_from_json(json.loads(f.read()))

	# load weights into new model
	model.load_weights("./model.h5")
	return model

def predict_steering(model, img):
	# 
	# Given a model and a raw input image, return the predicted steering angle.
	#
	X, _ = reshape_xy([normalize_image(process(img))], [0])
	return model.predict(X, batch_size=1, verbose=0)[0][0]/signal_scale


