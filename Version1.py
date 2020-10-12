# AI
# Assignment 4

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses

# load data
samplesubmission = pd.read_csv("nn-assignment/samplesubmission.csv")
testset = pd.read_csv("nn-assignment/testset.csv")
trainset = pd.read_csv("nn-assignment/trainset.csv")

# check that loded correctly
print(f'Head of samplesubmission: \n', samplesubmission.head())
print(f'Testset dim: ', testset.shape)
print(f'Trainset dim: ', trainset.shape)
print(f'Head of trainset: \n', trainset.head())

#split traningset into traning and validationset
trainset = trainset.sample(frac=1) # shuffle data
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
#print(images[0,:,:])
#print(images[0,:,:].shape)

# Normalize pixel values to be between 0 and 1
#train_images = train_images / 255
#valid_images = valid_images / 255


# plot first 25 train_images to check reshaping
plt.figure(figsize=(28, 28))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(train_images[i])
    plt.xlabel(train_labels[i])
plt.show()
print(f'First 25 labels: ', train_labels[0:25])

#create convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#add dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


print(test_acc)



#compile and train model


#The size of the output layer is equal to the number of classes.


