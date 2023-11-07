#QUESTION1

#Load the iris dataset from the sklearn library and create

from sklearn import datasets
datasets.load_iris()

#3 different machine learning classification models

from sklearn import linear_model
linear_model.LogisticRegression()

from sklearn import neighbors
neighbors.KNeighborsClassifier()

from sklearn import svm
svm.SVC()

#Use the standard 80:20 training/testing split to evaluate your model
from sklearn import model_selection
model_selection.train_test_split(train_size = 80, test_size = 20)

#Utilize a variety of different metrics to perform a thorough evaluation 
#of your various models to determine which model is best. 

from sklearn import iris_data

iris_data.data.shape()
len(iris_data.feature_names)
iris_data.target_names

#Write a brief summary of your conclusions determining which model is the best performer.
#The best performer is determined by the data shape and feature name evaluations, 
#as they offer proper information on the iris dataset. 

#QUESTION2

#Extract the sepal length and sepal width from the iris data

from sklearn import datasets
datasets.load_iris()
iris_data.feature_names

#Split the data in 135 training observations and 15 testing observations, 
#use random_state equals 0

from sklearn import model_selection
model_selection.train_test_split(train_size = 135, test_size = 15, random_state = 0)

#Create an interpolation predicting the sepal width given the sepal length

from scipy import interpolate
import numpy as np
interpol = interpolate.interp1d(sepal_length)
print(interpol)

#Create a regression model using the same data as the interpolation. 

from sklearn import linear_model
linear_model.LogisticRegression(interpol)

#Compare the two function approximations using the sum of the absolute value
#of the differences between the true values and the approximated values

from sklearn import metrics
metrics.mean_absolute_percentage_error(sepal_length, sepal_width)

#Write a brief explanation about which approximation you think is 
#better and why.

#I think that the sepal length approximation is better, due to the fact 
#that the approximated values have less error. 


 