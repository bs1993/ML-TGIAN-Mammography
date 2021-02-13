#Info

•Writtter's Name: Vassilis Carlis
•Date: 02.2021

An implementation for Machine Learning Course in the case of MSc Artificial Intelligence, NCSR Demokritos


#Table of Contents

•	Find Data in
•	Introduction
•	Goal
•	Features
•	Scatter Matrix
•	Data Pre-process
•	Classification Algorithms
•	Evaluation Metrics
•	Evaluation Visualisations
•	Run File


#Find the Data in

link: https://raw.githubusercontent.com/jbrownlee/Datasets/master/mammography.csv


#Introduction

At this implementation is used classification algorithms in order to predict breast cancer through mammography data. The data is stored in the mammography.csv file. In this file are contained 11.183 objects which identified from six features.


#Goal

The goal of this project is the development of a useful supportive tool for the medical staff.


#Features

•	area
•	grey_level
•	gradient_strength
•	flucturiation
•	contrast
•	low_order_moment



#Scatter Matrix

In order to acquire a macroscopic view for the correlation of the features, a scatter matrix among all features’ combination is printed.


#Data Pre-process

The following methods are used for the data pre-processing.
•	Normalization using MinMax method.
•	A method for the removal of correlated features.


#Classification Algorithms 

• Logistic Regression
•	K Nearest Neighbors
•	Support Vector Machines
•	Decision Trees
•	Naive Bayes


#Evaluation Metrics

The following metrics are utilized for the evaluation of algorithms

•	Accuracy

Furthermore, the following metrics are manually computed:

•	True Positive
•	False Positive
•	True Negative
•	False Negative
•	Precision
•	Recall
•	Specificity
•	F1 Score


#Evaluation Visualizations

Confusion Matrixes and ROC (Receiver Operating Characteristics) are visualized for the better evaluation of algorithms.


#Run File

For the run of the program is required a Python 3 IDE and the installation of the following libraries:

•	pandas
•	numpy
•	matplotlib	

Also, the file classification.py and the file mammography.csv must be stored in the same folder.


