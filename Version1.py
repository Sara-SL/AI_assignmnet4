# AI
# Assignment 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# load data
samplesubmission = pd.read_csv("nn-assignment/samplesubmission.csv")
testset = pd.read_csv("nn-assignment/testset.csv")
trainset = pd.read_csv("nn-assignment/trainset.csv")

# check that loded correctly
print(f'Head of amplesubmission: \n', samplesubmission.head())
print(f'Testset dim: ', testset.shape)
print(f'Trainset dim: ', trainset.shape)
print(f'Head of trainset: \n', trainset.head())

# extract labels from traningset
labels = np.array(trainset['label'])

# extract images from traningset & reshape
images = np.array(trainset.drop(['label'], axis=1))
images = np.reshape(images, (28000, 28, 28))
# print(images[0,:,:])
# print(images[0,:,:].shape)

# Normalize pixel values to be between 0 and 1
images = images / 255

# plot first 25 images to check reshaping
plt.figure(figsize=(28, 28))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(images[i])
plt.show()
print(f'First 25 labels: ', labels[0:25])



# #split trainset into traning - and validationset (70/30)
# trainset['split'] = np.random.randn(trainset.shape[0], 1)
# print(trainset['split'].head)
# msk = np.random.rand(len(trainset)) <= 0.7
# print(msk[1])
# train = trainset[msk]
# valid = trainset[~msk]
# print(train.head)
# print(train.shape)
# print(valid.shape)
# print(type(train))


# class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
