# Facial Emotion Recognition on fer2013
An accuracy of 68% on the validation data on the fer2013 is achieved using a model that was trained by generating new images from the data set.
## requirements
Python version: 3.7

The following python packages are required
* numpy
* openCV
* pandas
* matplotlib
* tensorflow/keras

## Script main.py
Requirements:
The file 'haarcascade_frontalface_default.xml' inside the repository folder is required to execute this script. 
The file can be downloaded e.g. from here: https://github.com/opencv/opencv/tree/master/data/haarcascades
The fer2013 data set has to be downloaded (e.g. https://www.kaggle.com/deadskull7/fer2013) and located inside the repository folder to execute this script.

This script can be executed afterwards using the command:
```python
python main.py
```
This script loads the 'main.h5' model by default and predicts images from the fer2013.csv data set.

### Change the loaded model
The model can be changed in line 116:
```python
model.load_weights('main.h5')
```

### Enable training with image generation
In order to train the model using the image generator line 107 has to be commented out and the loading in line 116 has to be commented:
```python
model_history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size , epochs=epochs ,batch_size=batch_size ,validation_data=(X_valid, y_valid))
```
and
```python
# model.load_weights('main.h5')
```
### Enable training without image generation
In order to train the model using the image generator line 107 has to be commented out and the loading in line 116 has to be commented:
```python
model_history = model.fit(X_train, y_train , epochs=epochs ,batch_size=batch_size ,validation_data=(X_valid, y_valid))
```
and
```python
# model.load_weights('main.h5')
```

## Script: predictImage.py
Requirements:
The file 'haarcascade_frontalface_default.xml' inside the repository folder is required to execute this script. 
The file can be downloaded e.g. from here: https://github.com/opencv/opencv/tree/master/data/haarcascades
This script can be executed to predict an image (e.g. testImage.png) by passing the image as an argument:
```python
python finalNet.py testImage.png
```
