# AI
# Assignment 4

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

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


#add dense layers on top


#compile and train model





