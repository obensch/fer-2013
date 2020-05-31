# import numpy for calculations and pandas to load CSV
import numpy as np
import pandas as pd
from matplotlib import pyplot

# Import tensorflow and keras to train the net
import tensorflow as tf
from tensorflow import keras 
from keras.preprocessing.image import ImageDataGenerator

#load csv
Data = pd.read_csv('fer2013.csv')

# load images to array
pixels = Data.pixels
pixels = pixels.str.split(expand=True)
img_arr = np.array(pixels).reshape(-1, 48, 48, 1).astype(int)

## Code block to equalize the histogram's of the data set
# startimage = 20

# fig, ax = plt.subplots(nrows=2, ncols=10, figsize=(25,25))
# for i in range(10):
#   ax[0][i].imshow(np.array(img_arr[startimage+i]), cmap='gray')
#   ax[0][i].set_title(i)

# for i in range(0,len(img_arr)):
#   img_arr[i] = cv.equalizeHist(img_arr[i])

# for i in range(10):
#   ax[1][i].imshow(np.array(img_arr[startimage+i]), cmap='gray')
#   ax[1][i].set_title(i)

# store labels to numpy array
img_labels = Data.emotion.values 

# split test data
X_valid, X_train = img_arr[:3000], img_arr[3000:]
y_valid, y_train = img_labels[:3000], img_labels[3000:]

# get image shapes
img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = 7

# normalize data
X_train = X_train / 255.
X_valid = X_valid / 255.

# output images
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
print("Input Shape:", img_width, img_height, img_depth, num_classes)

# set Seed
tf.random.set_seed(77)

# Build model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=(img_width, img_height, img_depth)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Flatten())	

model.add(keras.layers.Dense(128, activation="relu"))	
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.6))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.6))
model.add(keras.layers.Dense(num_classes, activation="softmax"))

# set comiler settings
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["acc"])

# Generate transformed images
datagen = ImageDataGenerator(
        zoom_range=0.2,          # randomly zoom into images up to 20% 
        rotation_range=20,       # randomly rotate images in the range to 10 degress 
        width_shift_range=0.1,   # randomly shift images horizontally up to 10%
        height_shift_range=0.1,  # randomly shift images vertically up to 10%
        horizontal_flip=True,    # randomly flip images horizontal
        vertical_flip=False)     # dont flip images vertical

batch_size = 128
epochs = 300

# fit model with Data Generator 
# model_history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epochs ,batch_size=batch_size ,validation_data=(X_valid, y_valid))

# fit model without Data Generator 
# model_history = model.fit(X_train, y_train , epochs=epochs ,batch_size=batch_size ,validation_data=(X_valid, y_valid))

# save model
# model.save("main.h5")

# Restore the weights
model.load_weights('main.h5')

np.random.seed(7)
fig = pyplot.figure(1, (14, 14))

labels = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

randomImages = []

for x in range(7):
    randomImages.append(np.random.choice(np.where(y_valid == x)[0], size=7))

correct=0    
k=0
for i in range(5):
  for label in range(7):
    px = X_valid[randomImages[label][i],:,:,0]
    k += 1
    ax = pyplot.subplot(7, 7, k)
    ax.imshow(px, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Emotion:{labels[label]} \nPredicted:{labels[model.predict_classes(px.reshape(1,48,48,1))[0]]}")
    pyplot.tight_layout()
    if(labels[label] == labels[model.predict_classes(px.reshape(1,48,48,1))[0]]):
      correct +=1

pyplot.show()
print(print(correct/k * 100))