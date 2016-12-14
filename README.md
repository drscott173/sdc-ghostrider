# udacity-SDC-ghostrider

SDC Project 3 - a deep learning network that learns to drive by watching you.

We seek to create a neural network that predicts a steering angle from an image,
where the image is a frame from a car racing videogame. A sample
image is seen below.

![Input Image](images/frame.png?raw=true "An input frame")

We obtain training images
by playing the video game and capturing 10 images per second, along with the
current values for the brake, steering, and acceleration.  We test the network
by feeding raw images into our network, extracting the steering angle, then transmitting
the angle into the game via a socket connection.

## Solution Design

Good programmers learn from great programmers.  Researchers at NVidia had created
[a deep neural network](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) 
for driving a car solely from camera input.  They biased training examples to prefer snapshots where turning is involved.  This was to avoid overfitting 
to the usual case of driving straight.  These images were separated into three YUV channels
and fed into a deep network.  

For pedagogical purposes, we've recreated their convolutional neural network in
Keras with a TensorFlow backend.  We've reduced the image size and chose black
and white vs. color images to train on a laptop CPU. 

## Model Architecture

![NVidia's CNN Model](images/network.png?raw=true "CNN architecture (courtesy NVidia)")

The input X to our model is a tensor of dimension [?, 80, 160, 1], where ? is the
number of 160x80 black and white images stored in row-major order.  The labels y to our
network are of dimension [?] where each value is the observed steering angle from
our simulated driving for the corresponding image.

We add a normalization up front to simplify fitting and improve 
performance of network optimization.  This normalization blocks out the top and
bottom of the image with zero's to focus on the driving area, 
then normalizes pixel values to 
the interval [-0.5, 0.5].

![Normalized Image](images/data.png?raw=true "A normalized image as input data")

Our model seen above feeds these normalized images into
a succession of 5 convolutional networks (CNNs) with RELU activation, followed
by a tapering of 3 fully connected layers.  The first two CNNs
also add a dropout filter that squashes half of the values randomly on every cycle
to prevent overfitting.  The layers are 

- An input layer of [?, 80, 160, 1] images
- A virtual "normalization" layer using code (see 'process' routine in model.py)
- 5x5 convolution with 24 filters, with a stride of 2, 50% dropout, RELU activation
- 5x5 convolution with 36 filters, with a stride of 2, 50% dropout, RELU activation
- 5x5 convolution with 48 filters, with a stride of 2, RELU activation
- 3x3 convolution with 64 filters, with a stride of 2, RELU activation
- 3x3 convolution with 64 filters, with a stride of 2, RELU activation
- A flattening layer
- A fully connected linear layer of 100 neurons, RELU activation
- A fully connected layer of 50 neurons, RELU activation
- A fully connected layer of 10 neurons
- An output layer of 1 neuron

## Training Dataset

We created a subdirectory "train" to hold our training images from the simulation.
Each directory within "train" holds an IMG directory with screen captures of the 
driving simulation.  The directory also contains a CSV file that lists image pathnames
followed by steering angle, throttle, and so forth.  We manually modified the 
pathnames to make them relative vs. absolute for improved sharing.

We wrote processing routines that convert the JPEG images into normalized
images and extract the driving angle from the CSV.  The images and angles are
stacked vertically to create a feature set X and a label set y.  We store
the array [X,y] in data.p within the directory.

For debugging purposes we also create animated videos of both the extracted
images as well as early experiments in identifying lines in the image.  The latter
were not used upon further review of the NVidia paper.

Using this format we then ran multiple simulations.  We captured 5 different
sessions of correcting steering when you drive off the "left" side of the road,
then 5 sessions of corrected steering when you drive off the "right" side of the
road, followed by "normal" driving as best we could in the center.   All told
we had nearly 25,000 examples.

We create a training set by extracting all these files into memory.  Once there,
we first extracted all the input examples with non-zero driving angle as these
are the "turning" examples from the paper.  We complement this with a random 
selection of "going straight" examples where the steering angle was zero.  We
balanced the dataset so that half were turning, half were straight.

## Training Process

We used a modified version of k-fold validation.  Since training of this complex
network was slow on a laptop, we would extract and shuffle a training set
from the original data every 5 epochs.  From this shuffled set we'd use 80 percent
for training with a batch size of 100, and 20 percent for evaluation.
We would also save the network and its
weights to allow us to resume our life (you know, to eat, or sleep).

We compared the predicted steering angle from our network against the desired
angle y.  If these differed, we'd calculate the squared error and adjust
the weights to reduce this error as we back-propagated through the network to
the inputs.  We used the Adam algorithm to attenuate both the learning rate
and delta applied to each weight or bias over time.

If you're curious, you can check out our 
[Python notebook](GhostDriver.ipynb) to see the handiwork as
we tried multiple algorithms and techniques to get this right.

## Simulation

Note:  Please load model.py and the "predict-steering(model,image)"
when evaluating the model.  We have updated "drive.py" to do this
correctly.

We ran 400 epochs over 16 hours of training.  The result showed early signs
of learning to ride the track!  It's a hair-raising ride, for sure.  The car is
able to stay on the road and navigate a few turns before veering off or
getting stuck.

With over 250,000 weights the network clearly neeeds (1) more data and (2) more
training time.  Yet this pedagogical example demonstrates that we're on the right
path and convergence feels plausible.

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

