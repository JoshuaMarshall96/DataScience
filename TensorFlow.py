#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 00:19:11 2019

@author: Joshua
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Images Of Hand Written Digits Between 0-9
mnist = tf.keras.datasets.mnist

#Fix Data To Training & Test Variables
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#Normalized Data Between 0-1 For Optimized Neural Network
X_train = tf.keras.utils.normalize(X_train, axis = 1)
X_test = tf.keras.utils.normalize(X_test, axis = 1)

#Using A Simple Feed-Forward Network To Avoid Unncessary Cycling
model = tf.keras.models.Sequential()
#Flatten Mutlidimensional Array Of Image Data For First Layer Of Network
model.add(tf.keras.layers.Flatten())
#Set Number Of Neurons & Activation Function On Second Layer
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
#Create Third Layer To Refine Results
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
#Output Layer Of 10 Neurons. Softmax Used To Deal With Probability Distribution
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

#Set How The Network Will Be Assessed, Networks Optimize Based On Reduction Of Loss So This Is Key
model.compile(optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metric = ["accuracy"])


#Fix The Model Generated To The Data
model.fit(X_train, Y_train, epochs = 3)


#Evaluate The Effectiveness Of The Model
val_loss, val_acc = model.evaluate(X_test, Y_test)
print(val_loss, val_acc)

#Save Model 
model.save("num_recog")

#Use Model To Make Predictions To Test Validity
new_model = tf.keras.models.load_model("num_recog")
predictions = new_model.predict([X_test])

#Access One Hot Array With Numpy To Show Prediction
print(np.argmax(predictions[0]))

#Show Image To Compare
plt.imshow(X_test[0])
plt.show()


