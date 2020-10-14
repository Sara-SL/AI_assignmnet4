# AI
# Assignment 4
# By: Sara Lundqvist and Leo Hoff von Sydow

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, metrics, optimizers

# load data
samplesubmission = pd.read_csv("nn-assignment/samplesubmission.csv")
testset = pd.read_csv("nn-assignment/testset.csv")
trainset = pd.read_csv("nn-assignment/trainset.csv")

# check that loded correctly
print(f'Head of samplesubmission: \n', samplesubmission.head())
print(f'Testset dim: ', testset.shape)
print(f'Trainset dim: ', trainset.shape)
print(f'Head of trainset: \n', trainset.head())

# split traningset into traning and validationset
trainset = trainset.sample(frac=1)  # shuffle data
train, valid = train_test_split(trainset, test_size=0.15)
print(train.head)
print(valid.head)

# extract labels from traningset
train_labels = np.array(train['label'])
valid_labels = np.array(valid['label'])

# extract pixelvalues from trainset and validationset & reshape to 28x28 arrays
train_images = np.array(train.drop(['label'], axis=1))
train_images = np.reshape(train_images, (23800, 28, 28))
valid_images = np.array(valid.drop(['label'], axis=1))
valid_images = np.reshape(valid_images, (4200, 28, 28))
test_images = np.array(testset)
test_images = np.reshape(test_images, (14000, 28, 28))
test_images = test_images.astype('float32')

# normalize pixel values to be between 0 and 1
train_images = train_images / 255
valid_images = valid_images / 255

# plot first 25 train_images to check reshaping
plt.figure(figsize=(28, 28))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(train_images[i], cmap=plt.cm.gray)
    plt.xlabel(train_labels[i])
plt.show()
print(f'First 25 labels: ', train_labels[0:25])

# reshape images to work as argument in models.fit()
train_images = np.reshape(train_images, (23800, 28, 28, 1))
valid_images = np.reshape(valid_images, (4200, 28, 28, 1))
test_images = np.reshape(test_images, (14000, 28, 28, 1))

# create model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # Downsamples the input
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))  # Helps to prevent overfitting
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

# compile and train model
model.compile(loss='sparse_categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=32, epochs=5, validation_data=(valid_images, valid_labels))

# evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# print accuracy
test_loss, test_acc = model.evaluate(valid_images, valid_labels, verbose=2)
print(f'Test accuracy: ', test_acc)

# predict labels on testset and save in csv file
imageID = np.arange(1, 14001)
pred = model.predict(test_images)
label = np.argmax(pred, axis=1)
df = pd.DataFrame({"ImageID": imageID, "Label": label})
df.to_csv("Submission.csv", index=False)
