# udacity-SDC-ghostrider

SDC Project 3 - a deep learning network that learns to drive by watching you.

We seek to create a neural network that predicts a steering angle from an image,
where the image is a frame from a
[car driving simulator](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip). A sample
image is seen below.

![Input Image](images/frame.png?raw=true "An input frame")

We obtain training images
by playing the video game and capturing 10 images per second, along with the
current values for the brake, steering, and acceleration.  We test the network
by feeding raw images into our network, extracting the steering angle, then transmitting
the angle into the game via a socket connection.

## TL; DR version

Alas, it works.  Amazing.  We've built a neural network from scratch
that learns to drive in an animated, 3D driving game.  The AI was
trained on one game then fed a completely new surrounding and did
just fine.

[![Final Result](https://img.youtube.com/vi/Wi1_rnNKB18/0.jpg)](https://www.youtube.com/watch?v=Wi1_rnNKB18)

I spent over two weeks tracking down an obscure bug.  I was copying the raw RGB image 
from the simulation to an array using "numpy.asarray".  However, this was broken on Python 3.5
on my Mac producing nearly completely white images.  My elaborate models seemed stuck
on a single output value.  Instead, "numpy.array()" casting was all I needed.

Lesson learned.  Always, and I do mean always, check the images on your data pipeline
to make sure memory is copied correctly with the correct transformations and bit order!

## Dependencies

This pedagogical example requires that the simulator be run in 640x480
resolution.  We also require a Python 3.x environment with the following packages:

- Anaconda Python Distribution
- numpy
- flask-socketio
- eventlet
- pillow
- h5py
- matplotlib
- pandas
- keras
- tensorflow

The GitHub distribution includes the sample images.  The generated "data.p" data files
must be generated on demand as the results are too large to store on GitHub.  This will
happen automatically the first time you train a model.

We include the following key files so you can get started quickly:

- model.json, a json file storing the keras convolutional network
- model.h5, weights after a dozen hours of training
- drive.py, the autonomous server, run with "python drive.py model.json"

## Solution Design

Good programmers learn from great programmers.

Researchers at NVidia had created
[a deep neural network](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) 
for driving a car solely from camera input.  They biased training examples to prefer snapshots where turning is involved.This was to avoid overfitting 
to the usual case of driving straight.  These images were separated into three YUV channels
and fed into a deep network.  

For pedagogical purposes, we've recreated their convolutional neural network in
Keras with a TensorFlow backend.  We have inserted dropout regularization after every
two convolutional layers.  We also treated the last four fully connected layers
as a "control nozzle" that would combine the 1152 flattened, activated logits
in non-linear ways.  These last layers do not have activation.


## Model Architecture

![NVidia's CNN Model](images/network.png?raw=true "CNN architecture (courtesy NVidia)")

The input X to our model is a tensor of dimension [?, 160, 320, 3], where ? is the
number of 320x160 color images stored in row-major order, one plane for
each color.  Images are expected to use the YUV color space which has been known 
to better distinguish gradient changes in light intensity, which are crucial for edge
detection.  We also include a simple convolution layer up front that combines the 
channel values and surrounding pixels in a simple linear fashion.  Color has been
shown to have value in navigation.  This lets the model figure out how to modify
that color.

The labels y to our
network are of dimension [?] where each value is the observed steering angle from
our simulated driving for the corresponding image.

We add a normalization up front to simplify fitting and improve 
performance of network optimization.  This normalization blocks out the top and
bottom of the image with zero's to focus on the driving area, 
then normalizes pixel values to 
the interval [-0.5, 0.5].

![Normalized Image](images/data.png?raw=true "A normalized image as input data")

The model seen above was first created by NVidia for a deep learning approach to
detecting steer purely from images. 

Our model is a slight modification to this, inserting pooling and dropout layers,
and using 2x2 max pooling instead of larger 2x2 strides in the middle of the network.
This proved quite useful in adapting to unseen conditions in the second test
road. The layers are 

- An input layer of [?, 160, 320, 3] YUV images
- A virtual "normalization" layer using code (see 'process' routine in model.py)
- A 1x1 convolution with 3 filters, linear output for color modification
- 5x5 convolution with 24 filters, with a stride of 2, 50% dropout, ELU activation
- 5x5 convolution with 36 filters, 50% dropout, ELU activation
- A 2x2 Max Pooling layer in lieu of a 2x2 stride as seen with NVidia
- A 50% dropout layer to avoid overfitting
- 5x5 convolution with 48 filters, RELU activation
- 3x3 convolution with 64 filters, RELU activation
- A 2x2 Max Pooling layer in lieu of a 2x2 stride as seen with NVidia
- A 50% dropout layer to avoid overfitting
- 3x3 convolution with 64 filters, with a stride of 2, ELU activation
- A flattening layer
- A fully connected linear layer of 100 neurons
- A fully connected layer of 50 neurons
- A fully connected layer of 10 neurons
- An output layer of 1 neuron

The output layer yielded our predicted steering angle. This was compared against
the desired angle.  An Adam optimzer was used for back propagation, using the mean
squared error between the desired and predict angle as a cost function during
optimization.

## Training Dataset

We created a subdirectory "train" to hold our training images from the simulation.
Each directory within "train" holds an IMG directory with screen captures of the 
driving simulation.  The directory also contains a CSV file that lists image pathnames
followed by steering angle, throttle, and so forth. 

We wrote processing routines that convert the JPEG images into normalized
images and extract the driving angle from the CSV.  The images and angles are
stacked vertically to create a feature set X and a label set y.  This creates 
an input array [X,y] for the neural networks.

We captured four different
sessions driving slowly at 10mph with a keyboard control.  The joystick
would have been far easier!  The sessions were:
 
- One lap around the track, being careful to stay in the center and hit turns slowly
- One lap going the opposite way
- One lap where we constantly veered off the main road to the left, turned on the camera,
recorded us getting back into the road center, going forward, turning off the camera,
and repeating.
- One lap doing the "off road" trick but on the right

## Data Augmentation

We augmented captured data with image filters to simulate other
situations and essentially create an "infinite" dataset.  The filters
were:

- Brightness filter, making the scene randomly darker or brighter
- Shadow filter, layering random gray shadows over the image
- Translation filter, randomly shifting the image
- Reverse filter, randomly flipping the image horizontally 

The translation and reverse filters required slight modifications to the 
steering angle, which we achieved through trial and error.

## Training Process

We first counted all the captured images, from all subdirectories of the
train directory.  This became our "epoch" size.

We created a Python generator, an object that continually emits one image and
one steering angle from these captured images.  The generator would first randomly
shuffle the subdirectories (left, right, normal, reverse). Once in a directory,
the generator would shuffle the images and choose 95% for training, 5% for validation.
Every image was randomly augmented.  The augmentation process would often leave
an image untouched, too.

The generator initialy prefers steering angles above 0.1 and skips all others.
As time progresses,
the generator introduces more subtle angles to avoid bias towards 0 (which dominates
the data) with exponential decay, just like the learning rate in Adam optimizers.

We created a second, batch generator that fed off the individual image and steering
angle generator.  This is what we used for training.  The batch generator would
queue up 32 images, create a 32-deep pair of X,y values, then feed this into 
the newtwork.

We compared the predicted steering angle from our network against the desired
angle y.  If these differed, we'd calculate the squared error and adjust
the weights to reduce this error as we back-propagated through the network to
the inputs.  We used the Adam algorithm to attenuate both the learning rate
and delta applied to each weight or bias over time.

If you're curious, you can check out our 
[Python notebook](GhostDriver.ipynb) to see the handiwork as
we tried multiple algorithms and techniques to get this right.

## Simulation

Note:  We modified drive.py so that our throttle would mimic cruise control
at roughly 10mph, hitting the gas or slowing down as needed.  This helped us
with hills.

With over 250,000 weights the network was able to successfully complete the 
first track and would run for hours without going off road.  As we sped the
car up, our training data was inaccurate as the angles were too sharp for
faster driving.  We attenuated the predicted angle a bit based on speed.

We ran 100 epochs over 4 hours of training on a CPU.  The network demonstrates 
that we're on the right path and convergence feels plausible.

Many students of this course observed that training seemed to "get stuck" after
a half dozen epochs.  We saw the same behavior, which at first was highly
frustrating -- 60 percent accuracy was barely better than pure chance.  Further,
a stupid algorithm that would fix driving at 0 outperformed the network.

It then occurred to us that the input data was heavily biased toward 0. The
accuracy was directly correlated to the percentage of "drive straight"
training examples.   The
network converged on 0 and was unable to capture the statistical outliers.
However, those outliers were in fact what we needed.  Fixing the dataset to
prefer turns with non-zero steering angles got our network to learn again.

