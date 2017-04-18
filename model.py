import csv
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Input, Conv2D, MaxPooling2D, Dropout, Reshape
from keras.optimizers import Adamax, Adam

###Load data
#Load csv
def load_samples(file):
    lines = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

#Compute number of samples
def compute_num_samples(num, flip=False, left_rigth_camera=False):
    samples = num
    if flip:
        samples = samples + num
    if left_rigth_camera:
        samples = samples + num*2    
    return samples

#Generator of data for real time augmentation: flip and left/rigth cameras
def generator(path, samples, batch_size=32, flip=False, left_rigth_camera=False):
    #Open image
    def open_image(path, line):
        filename = line.split('/')[-1]    
        filename_path = (path + 'IMG/' + filename)        
        image = cv2.imread(filename_path, cv2.IMREAD_COLOR)
        #Adjust color space because OpenCv uses BGR and simulator RGB
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #Crop the image
        image = image[70:140, 0:320]
        #Resize to 200x66 pixels
        image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)
        return image

    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for line in batch_samples:
                image = open_image(path, line[0])
                images.append(image)
                measurement_center = float(line[3].replace(',','.'))
                measurements.append(measurement_center)
                if flip:                    
                    image = cv2.flip(image, 1)
                    images.append(image)
                    measurements.append(measurement_center*-1) 
                if left_rigth_camera:
                    correction = 0.25 
                    measurement_left  = min(measurement_center + correction,1) #min 1
                    measurement_right = max(measurement_center - correction,-1) #max -1
                    #Left
                    image = open_image(path, line[1])
                    images.append(image)
                    measurements.append(measurement_left)
                    #Right
                    image = open_image(path, line[2])
                    images.append(image)
                    measurements.append(measurement_right)
            X_train = np.array(images)                        
            y_train = np.array(measurements)            
            yield sklearn.utils.shuffle(X_train, y_train)

#Get sample files (csv files)
samples1 = load_samples('videos/merge/new1.csv')
samples2 = load_samples('videos/merge/new2.csv')
samples3 = load_samples('videos/merge/new3.csv')
samples4 = load_samples('videos/merge/driving_log2.csv')
samples5 = load_samples('videos/merge/slow_log1.csv')
samples6 = load_samples('videos/merge/corner_log3.csv')
samples7 = load_samples('videos/merge/driving_log5.csv')
samples8 = load_samples('videos/merge/driving_log6.csv')
samples = samples1 + samples2 + samples3 + samples4 + samples5 + samples6 + samples7 + samples8

#Split dataset into training and validation 
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Recompute the number of samples because of the augmented images (flip, left and rigth images)
train_samples_size = compute_num_samples(len(train_samples), flip=True, left_rigth_camera=True)
validation_samples_size = compute_num_samples(len(validation_samples), flip=True, left_rigth_camera=True)

#Get Generators for training and validation
#Batch size depends on the augmented images (256 x 4 = 1028).
train_generator = generator('videos/merge/', train_samples, batch_size = 256, flip=True, left_rigth_camera=True)
validation_generator = generator('videos/merge/', validation_samples, batch_size = 256, flip=True, left_rigth_camera=True)

#Image input shape
input_shape = (66, 200, 3)

###Image Pre-Processing
#Normalization
normalize = Lambda(lambda x: ((x - 125.0) / 255.0), input_shape=input_shape, output_shape=input_shape)

###Nvidea Convolution Neural Network
model = Sequential()
model.add(normalize)
#Convolutions Layers
model.add(Conv2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Dropout(0.2))
#Control Layers
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
#Output 
model.add(Dense(1, activation='tanh'))

##Training
adam = Adamax()
model.compile(loss='mse', optimizer=adam)
model.summary()
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=train_samples_size,
                                     validation_data=validation_generator,
                                     nb_val_samples=validation_samples_size,
                                     nb_epoch=5,
                                     verbose=1)

#save model
model.save('model_track2.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png', dpi=100)
plt.show()



