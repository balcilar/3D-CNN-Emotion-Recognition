# Volumetric CNN For Emotion Recognition

3D point cloud become easily available within new technology devices. For example with IPhoneX, we can obtain 3D image of the face. 3D images can be represent in two way. One is within multi view of the object from different angle of view. In another way, using IR based cameras such as Kinect or IphoneX face recognition camera, we can obtain 3D point cloud in one shot. In that research, we proposed 3D convolutional Neural Network on recognition human face emotion class which we have 50 different person’s 7 different face emotion 3D point cloud images.

## Dataset

We have 50 subjects' 3D point cloud image under 7 different emotion which are angry, disgust, fear, happy, sad, surprise and neutral. So that means we have totally 50x7=350 different emotion in our dataset. We briefly divide first 40 subject’s data as train set and last 10 subject’s data as test set. In our dataset, each emotion image consist of 1220 number of x,y,z point triple coordinate. Thanks goodness of our concerned dataset, all 3D images are all in the same orientation and all normalized already. That is why we do not need to generate different scaled and oriented data to feed the CNN to get it generalize for all kind of test images. Here is the 7 different emotion of first subject's data.

![Sample image](Output/faces.jpg?raw=true "Title")

## Inputs

Instead of using 1220x3=3660 value directly, we selected to get these 1220 points in 3D 24x24x24 voxel matrix. To do that first we normalize each x,y and z coordinate in range of [0-23]. And we got it in integer form where we will use it as an index of volumetric matrix. To create volumetric matrix, we calculate every 1220 data points position in x,y,z coordinate into new index in 0-23. Let us called new index as ix,iy,iz. Then we accumulate the  voxel of Volumetric matrix’s ix,iy,iz by 1. As a result of all 1220 data points we accumulate corresponding voxel in volumetric matrix and finally we obtain final volumetric presentation of 3d point cloud. But note that if two or more x,y,z point drop in the same voxel, that the corresponding value at volumetric matrix become more than 1. So at the last stage, we divide all element in volumetric matrix to its own maximum value to get it in range of [0-1]. As a result we now have 24x24x24 matrix and its value in range of 0-1 as input. That means for training set all data is 280x24x24x24, for test data is in 70x24x24x24 dimension.

## Architecture

In our design we prefer to use 2 convolutional layer which is 32 number of 3D convolution matrix for given input. All two 3d Convolutional layer followed by normalization layer and maxpool layer. After convolution and maxpool layers, we get all output in line order (get it flat) and feed the fully connected later which has 128 number of neuron. Then we add drop out layer with 0.5 param to increase the ability of generalizability of the network. Finally it followed by 7 number of dense layer for classification, since or problem has 7 different number of classes. Here is the summarization of our model. Here is our architecture.

![Sample image](Output/cnn.jpg?raw=true "Title")

## Results

To train our model we prefer adam optimization algorithm with default parameter. We selected categorical cross entropy for loss function, since out problem is multi class classification problem. We performed training phase in 20 epochs and 16 batch size. Finally we reach %85 accuracy on our test case.

## Run
To run given code please launch cnnX.py source by followings.
```
$ python cnnX.py
```


