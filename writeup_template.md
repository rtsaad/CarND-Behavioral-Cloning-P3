# Self Driving Car Behavioral Cloning

This project consists of the development and training of a Convolution Neural Network (CNN) to clone human behavior for self-driving purpose. It uses the Udacity Car Simulator for training, test and validation. 

The main goals for this project is to develop a End-To-End Neural Network that gets as input the video from a camera positioned in front of the car and outputs the car steering angle. Figure 1 depicts the Udacity Simulator using the CNN to drive the car, the top right shows the image collected by the camera in front of the car. 

![alt text][image1]

## 1.Access 

The source code for this project is available at [project code](https://github.com/otomata/CarND-Behavioral-Cloning-P3).

## 2.Files

The following files are part of this project:
* model.py:   Contains the Convolution Neural Network implemented using Keras framework, together with the pipeline necessary to train and validate the model;
* model.h5:   Neural Network weights saved during training (model.py);
* driver.py:  Script that connects the NN with the Udacity Car Simulator for testing and validation;
* videos:     Data source used for training and validation.
* run1.mp4:   Video of the car driving autonomously on track one.

### Dependency

This project requires the [Udacity Simulator](https://github.com/udacity/self-driving-car-sim) in order to use the Convolution Neural Network to drive the car autonomously.

[//]: # (Image References)

[image1]: simulator.png "NN Driving the car"
[image2]: (original.jpg | width=100) "Recovery Image"
[image3]: (flip.png) "Flipped Image"
[image4]: (crop.png) "Croped Image"
[image5]: (resize.png) "Resized Image"
[image6]: loss1.png "Loss Chart" 

## 3.How to use this project

### Training the Model

The Convolution Neural Network provided with this project can be trained by executing the following command:

```sh
python model.py
```

### Driving Autonomously
To drive the car autonomously, this project requires the [Udacity Simulator](https://github.com/udacity/self-driving-car-sim). After installing and starting the simulator, the car can be driven autonomously around the track by executing the following command:
```sh
python drive.py model.h5
```

## 4.Training Data 

The data set used to train the model was obtained from the Udacity Simulator. The procedure to collect data followed an empirical process. First, we have driven three laps, in one direction only, at the first track  trying to stay as much as possible aligned at the center of the road. After this first batch of data collection, we tried to train the model. From the behavior shown by our car, we have observed that our model failed on the sharp corners where the side lanes are blurred. From this observation, we  collected more data concerning these corners, which are the corners after the bridge. 

The recovery from sides was achieved using the left and right camera images by adding a "correction angle" to the steering to force the car to regain the center of the road. Finally, the data collection was also augmented by flipping the center camera image otherwise the car would have a tendency to turn to the left because we collected data following only one direction. All this together proved sufficient to achieve the smooth steering shown in our results ([video](https://github.com/otomata/CarND-Behavioral-Cloning-P3/blob/master/run1.mp4)).

Our final data set had XXX number of pictures. During training, we shuffled the dataset at each epoch and divided it in 80% for training and 20% for validation, to analyze if the model is over fitting or under fitting (model.py lines 80-101).

## 5.Image Pre-Processing (Generators)

Pre-processing the data set is an important step not only to reduce the training time but also to achieve more favorable results. For every image, we first change the color space from BGR, which is the opencv color space, to RBG, which is the simulator color space. Then, we crop the image to remove areas that are not important for the model. Finally, we resize the to the final size of 200x66 pixels. Figure 2, 3, 4 and 5 depict every step of our pre-processing functions (model.py lines 13-79). 

![alt text][image2]

![alt text][image3]
![alt text][image4  | width=100]
![alt text][image5]


It is important to mention that all these pre-processing functions are encapsulated into a python generator to be executed online with the training algorithm. Python generator is an efficient way to save memory by pre-processing these images only when requested and without hold all of then in the working memory at the same time. The generator  operates in batches of small predefined sizes.


## 4.Neural Network Architecture

The Convolution Neural Network implemented for this project follows the [Nvidea End-To-End Deep Network for self driving cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). The network consists of 5 convolution layers followed by 5 fully connected layers. The input is a RGB image (of three channels) of 66x200 pixes -- which is normalized by a lambda keras function -- and the output is the value of the hyperbolic tangent activation function. In addition, we have augmented this network with four additional dropout layers to cope with the over fitting. We started by adding dropouts to the control layers (flatten layer). This dropout layer alone reduced significantly the overfiting but not enough for smooth driving. So, we started to add more dropouts for the convolution layers that are closer to the flatten layer because they have a more concrete notion of the image.  Figure 6 presents the network architecture (model.py lines 106-130).

| Layer        		|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input        		| 66x200x3 RBG image   				| 
| Normalization 	| Lambda, outputs 66x200x3			|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 31x98x24 	|
| RELU					|				|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 14x47x36 	|
| RELU					|                               |
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 5x22x48 	|
| RELU					|		                |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 3x20x64 	|
| RELU			|						|
| Dropout	        | Training probability of 20% 			|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 1x18x64 	|
| RELU			|						|
| Dropout	        | Training probability of 20% 			|
| Flatten		| 1164 neurons       				|
| RELU			|						|
| Dropout	    	| Training probability of 50%     		|
| Fully connected	| output 100       				|
| RELU			|						|
| Dropout	    	| Training probability of 50%     		|
| Fully connected	| output 50       				|
| RELU			|						|
| Fully connected	| output 10					|
| RELU			|						|
| Fully connected	| output 1					|
| TANH			|						|

## 6.Training and Model parameter tuning

The training algorithm employs the Adam optimizer algorithm with default values. The rest of the algorithm follows a common approach for regression problem which is to minimize the Mean Square Error (loss function). The values for the number of EPOCHS (20) and batch size (256*4=1024) were defined empirically. Another important parameter set during training was the "correction angle" mentioned at the pre-processing section. This angle is used to for the left and right cameras to force the car to return to the center of the road. From our empirical tests, we have set this angle to 0.25. Figure 7 presents the loss function for training and validation data sets (model.py lines 132-157). 

![alt text][image6]

## 7.Results (Simulation)

The simulation presented in our [video](https://github.com/otomata/CarND-Behavioral-Cloning-P3/blob/master/run1.mp4) demonstrates that our model was able to meet the safety requirements. Our car drives smooth around the first track without leaving the road.


