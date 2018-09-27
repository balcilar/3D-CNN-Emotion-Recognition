# Volumetric CNN For Emotion Recognition

3D point cloud become easily available within new technology devices. For example with IPhoneX, we can obtain 3D image of the face. 3D images can be represent in two way. One is within multi view of the object from different angle of view. In another way, using IR based cameras such as Kinect or IphoneX face recognition camera, we can obtain 3D point cloud in one shot. In that research, we proposed 3D convolutional Neural Network on recognition human face emotion class which we have 50 different person’s 7 different face emotion 3D point cloud images.

## Dataset

We have 50 subjects' 3D point cloud image under 7 different emotion which are angry, disgust, fear, happy, sad, surprise and neutral. So that means we have totally 50x7=350 different emotion in our dataset. We briefly divide first 40 subject’s data as train set and last 10 subject’s data as test set. In our dataset, each emotion image consist of 1220 number of x,y,z point triple coordinate. Thanks goodness of our concerned dataset, all 3D images are all in the same orientation and all normalized already. That is why we do not need to generate different scaled and oriented data to feed the CNN to get it generalize for all kind of test images. Here is the 7 different emotion of first subject's data.

![Sample image](Output/faces.jpg?raw=true "Title")
