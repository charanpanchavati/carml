""""
The code being modified is from: https://blog.coast.ai/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a
With admiration for and inspiration from:
    https://github.com/dolaameng/Udacity-SDC_Behavior-Cloning/
    https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    https://www.reddit.com/r/MachineLearning/comments/5qbjz7/p_an_autonomous_vehicle_steering_model_in_99/dcyphps/
Accompanies the blog post at https://medium.com/@harvitronix/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a
"""
import os.path
import json
import csv, random, numpy as np
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift


NUM_OUTPUTS = 2
SPEED = 1.0
def model(load, shape, checkpoint=None):
    """Return a model from file or to train on."""
    if load and checkpoint: return load_model(checkpoint)

    conv_layers, dense_layers = [32, 32, 64, 128], [1024, 512]
    
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='elu', input_shape=shape))
    model.add(MaxPooling2D())
    for cl in conv_layers:
        model.add(Convolution2D(cl, 3, 3, activation='elu'))
        model.add(MaxPooling2D())
    model.add(Flatten())
    for dl in dense_layers:
        model.add(Dense(dl, activation='elu'))
        model.add(Dropout(0.5))
    model.add(Dense(NUM_OUTPUTS, activation='linear'))
    model.compile(loss='mse', optimizer="adam")
    return model
    
def get_X_y(data_file):
    """Read the log file and turn it into X/y pairs. Add an offset to left images, remove from right images."""
    X, y = [], []
    steering_offset = 0.4
    with open(data_file) as fin:
        for _, left_img, right_img, steering_angle, _, _, speed in csv.reader(fin):
            if float(speed) < 20: continue  # throw away low-speed samples
            X += [left_img.strip(), right_img.strip()]
            y += [float(steering_angle) + steering_offset, float(steering_angle) - steering_offset]
    return X, y

def process_image(path, steering_angle, augment, shape=(120,160)):
    """Process and augment an image."""
    image = load_img(path, target_size=shape)
    
    if augment and random.random() < 0.5:
        image = random_darken(image)  # before numpy'd

    image = img_to_array(image)
        
    if augment:
        image = random_shift(image, 0, 0.2, 0, 1, 2)  # only vertical
        if random.random() < 0.5:
            image = flip_axis(image, 1)
            steering_angle = -steering_angle

    image = (image / 255. - .5).astype(np.float32)
    return image, steering_angle

def random_darken(image):
    """Given an image (from Image.open), randomly darken a part of it."""
    w, h = image.size

    # Make a random box.
    x1, y1 = random.randint(0, w), random.randint(0, h)
    x2, y2 = random.randint(x1, w), random.randint(y1, h)

    # Loop through every pixel of our box (*GASP*) and darken.
    for i in range(x1, x2):
        for j in range(y1, y2):
            new_value = tuple([int(x * 0.5) for x in image.getpixel((i, j))])
            image.putpixel((i, j), new_value)
    return image

def _generator(batch_size, X, y):
    """Generate batches of training data forever."""
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            sample_index = random.randint(0, len(X) - 1)
            sa = y[sample_index]
            image, sa = process_image(X[sample_index], sa, augment=False)
            batch_X.append(image)
            batch_y.append(sa)
        yield np.array(batch_X), np.array(batch_y)

#------------------

# NEW
def read_donkey_input(path):
    """ read dockey path """
    import glob
    X, y = [], []
    steering_offset = 0.4

    all_records = glob.glob('%s*.json'%path)
    #print (all_records)
    for jfile in all_records:
        if os.path.basename(jfile) == 'meta.json':
            continue
        data = json.load(open(jfile))
        ifile = data['cam/image_array']
        img_file = os.path.join(os.path.dirname(jfile),ifile)
        throttle = data['user/throttle']
        steering = data['user/angle']
        X += [img_file]
        y.append([float(steering), SPEED])
    return X, y

# NEW
def train1():
    """Load our network and our data, fit the model, save it."""
    net = model(load=False, shape=(120, 160, 3))
    X, y = read_donkey_input('./log_train/')
    x_val, y_val = read_donkey_input('./log_val/')
    #print y
    #exit(0)
    net.fit_generator(_generator(256, X, y), samples_per_epoch=100, validation_data=_generator(256, x_val, y_val), validation_steps=10, nb_epoch=3)
    x_test, y_test = read_donkey_input('./log_test/')
    net.save('./mypilot.h5')
    scores = net.evaluate_generator(_generator(256, x_test, y_test),10)
    #score = net.evaluate(x_test, y_test, verbose=0)
    #print score

if __name__ == '__main__':
    train1()
