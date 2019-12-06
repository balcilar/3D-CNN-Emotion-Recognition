import keras
from keras.layers import Input, Dense, Dropout, Flatten, Conv3D, MaxPool3D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from faces import face_samples
import numpy as np

# read dataset
# get point cloud in 24x24x24 voxel

num_classes = 7 
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


a=face_samples.keys()

X=np.zeros((len(a)*num_classes,24,24,24,1))
Y=np.zeros((len(a)*num_classes,num_classes))

#  READ IN iPHONE X DATA AND SHAPE

# create vortex 3d matrix
id=0
for xx in a:
    sub=face_samples[xx]
    emid=0
    for em in emotions:
        dat=sub[em]
        x=np.array(dat['x'])
        y=np.array(dat['y'])
        z=np.array(dat['z'])
        x= np.array(23.99*(x-x.min())/(x.max()-x.min()),dtype=np.int)
        y= np.array(23.99*(y-y.min())/(y.max()-y.min()),dtype=np.int)
        z= np.array(23.99*(z-z.min())/(z.max()-z.min()),dtype=np.int)        
        for i in range(0,len(x)):
            X[id,x[i],y[i],z[i],0]+=1
        X[id,:,:,:,0]/=X[id,:,:,:,0].max()
        Y[id,emid]=1
        emid+=1
        id+=1
print(X.shape)
print(Y.shape)



# assign first 40 subject (40*7=280 emotions) as train, last 10 subject (10*7=70 emotions) as test

X_train=X[0:280,:,:,:]
X_test=X[280:,:,:,:]
Y_train=Y[0:280,:]
Y_test=Y[280:,:]

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)



#  CREATE MODEL OF CHOICE
model = Sequential()
model.add(Conv3D(32, (5, 5, 3), activation='relu',input_shape = (24, 24, 24,1 )))
model.add(BatchNormalization())
model.add(MaxPool3D((3, 3, 2), strides=(2, 2, 2)))

model.add(Conv3D(32, (3, 3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool3D((2, 2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer = "adam", loss = 'categorical_crossentropy' , metrics = ['accuracy'])


# print out defined model
model.summary()


# train model on train set and evaluate on test set in every epoch
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=16)


# save model , precit test case
model.save('model_4.h5')
yhat = model.predict(X_test)


# calculate accuracy on test case
acc= (yhat.argmax(axis=1)==Y_test.argmax(axis=1)).mean()
print(acc)
