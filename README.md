# growthlinkproject

TABLE OF CONTENTS 

1.ABSTRACT  

2.INTRODUCTION  

3.DATA SET

3.1 What is a Dataset 

3.2 DATA SET COLUMN DESCRIPTION 

4.DATA ANALYSIS

5.PRE-REQUISITE LEARNING                                                                                                            
5.1 INTRODUCTION TO PYTHON

5.2 INTRODUCTION TO MACHINE LEARNING 

5.3 INTRODUCTION TO DATA SCIENCE

5.4 PYTHON FOR DATA SCIENCE 


6.LITERATURE SURVEY   

7.SOFTWARE REQUIREMENT      

8.METHODOLOGY                                                                                                                              
8.1 MACHINE LEARNING PROCESS

8.2 What are Bayesian Networks? 

8.3 NAIVE BAYES WORK FLOW 

8.4 NAIVE BAYES CLASSIFIER ALGORITHM                                                                                                                                

ABSTRACT


The sinking of the RMS Titanic is without a doubt one of the most infamous and horrific tragedies 
in history. Out of 2224 passengers and crew, the Titanic tragically sank on her maiden voyage and 
in the early hours of April 15, 1912, after colliding with an iceberg, killing almost 1502 of them. 
This made the ship one of the deadliest commercial ships in history up to that point. The laws 
governing ship safety have been toughened as a result of the horrific disaster that shocked the 
globe and caused it to feel profoundly sorry and scared. Thomas Andrews, the architect of the 
building, died in the disaster. It was a grim realisation following the sinking of the Titanic that 
certain persons had a higher chance of surviving than others. Child and mother priority had been 
given top priority. The Titanic was a prime example of the Titanic's time, which was the beginning 
of the 20th century and marked a severe division in socioeconomic groups. Exploratory data 
analytics (EDA) is utilised at the beginning and used to find facts that have been hidden or 
previously unknown in the existing data collection. 
This project aims to predict the survival of passengers aboard the Titanic using the Naive Bayes 
classifier algorithm. The dataset used in this project contains information about Titanic passengers, 
such as their Age, Sex, Pclass, and other relevant features. By training a Naive Bayes classifier on 
this data, we can predict whether a given passenger would have survived the Titanic disaster. 

INTRODUCTION

The inevitable development of technology has both facilitated our life and brought some 
difficulties with it. One of the benefits brought by the technology is that a wide range of data can 
be obtained easily when requested. However, it is not always possible to acquire the right 
information. Raw data that is easily accessed from the internet sources alone does not make sense 
and it should be processed to serve an information retrieval system. In this regard, feature 
engineering methods and machine learning algorithms are plays an important role in this process. 
The aim of this study is to get as reliable results as possible from the raw and missing data by using 
machine learning and feature engineering methods. Therefore one of the most popular datasets in 
data science, Titanic is used. This dataset records various features of passengers on the Titanic, 
including who survived and who didn't. It is realized that some missing and uncorrelated features 
decreased the performance of prediction. For a detailed data analysis, the effect of the features has 
been investigated. Thus some new features are added to the dataset and some existing features are 
removed from the dataset. 
Using data provided by www.kaggle.com , our goal is to apply Naive_Bayes Technique to 
successfully predict which passengers survived the sinking of the Titanic. Features like Ticket, 
Age, Sex, and Pclass will be used to make the predictions.        
Machine learning[8] means the application of any computer-enabled algorithm that can be applied 
against a data set to find a pattern in the data. This encompasses basically all types of data science 
algorithms, supervised, unsupervised,segmentation, classification, or regression". few important 
areas where machine learning can be applied are Handwriting Recognition, Language Translation, 
Speech Recognition, Image Classification, Autonomous Driving. Some features of machine 
learning algorithms can be observations that are used to form predictions for image classification, 
the pixels are the features, For voice recognition, the pitch and volume of the sound samples are 
the features and for autonomous cars, data from the cameras, range sensors, and GPS.   
Naive Bayes is a classification algorithm which is based on Bayes theorem with strong and naïve 
independence assumptions. It simplifies learning by assuming that features are independent of 
given class.This paper surveys about naïve Bayes algorithm, which describes its concept, hidden 
naïve Bayes, text classification, traditional naïve Bayes and machine learning. Also represents 
augmented naïve Bayes by examples. And at the end some applications of naïve Bayes and its 
advantages and disadvantages has discussed for a better understanding of the algorithm.

DATA SET 

What is a dataset: 

A data set, as the name suggests, is a collection of data. In Machine Learning projects, we need a 
training data set. It is the actual data set used to train the model for performing various actions. 
Here, in this case, we will be using a dataset available on the internet. One can find various such 
datasets over the internet.The dataset that I’ve used in my code was the data available on Kaggle. 

DATA SET COLUMN DESCRIPTION 

The original data has been split into two groups : training dataset(80%) and test dataset(20%).The 
training set is used to build our machine learning models. The training set includes our target 
variable, passenger survival status along with other independent features like Sex, Fare, and Pclass. 
The test set should be used to see how well our model performs on unseen data. The test set does 
not provide passengers survival status. We are going to use our model to predict passenger survival 
status. The test set should be used to see how well your model performs on unseen data. For the 
test set, we do not provide the ground truth for each passenger. It is your job to predict these 
outcomes. For each passenger in the test set, use the model we trained to predict whether or not 
they survived the sinking of the Titanic. 

• Pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd) 
• Survived: Survival (0 = No; 1 = Yes) 
• Name: Name 
• Sex: Sex 
• Age: Age 
• SibSp: Number of siblings/spouses aboard 
• Parch: Number of parents/children aboard 
• Fare: Passenger fare (British pound) 
• Embarked: Port of embarkation (C = Cherbourg; Q = Queenstown; S = 
Southampton)

Age 

Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5 

SibSp 

The dataset defines family relations in this way: 
• Sibling= brother, sister, stepbrother, stepsister 
• Spouse= husband, wife (mistresses and fiancés were ignored)

Parch 

The dataset defines family relations in this way:
• Parent= mother, father 
• Child= daughter, son, stepdaughter, stepson 
Some children traveled only with a nanny, therefore parch=0 for them.

DATA ANALYSIS

In order to prepare our data for training in our Naïve Bayes classifier, we remove or replace blank 
values and determine bin sizes for each feature. For instance, fare can range a large number of 
values, so we group fares together. The same is done for cabin data since the data is divided into 
cabin sections (A,B,C,D). A similar grouping is done to the ticket data. For our Naive_Bayes, we 
do not need to bin values together. Instead we simply turn all the values into numerical values. In 
order to do this, we’ll interpret the bit representation of strings and characters as float represented 
numbers. 

PRE-REQUISITE LEARNING 

A general-purpose, high-level programming language, Python has gained popularity recently. It 
enables programmers to write code in fewer lines, something that is not achievable in other 
languages. Python programming is notable for its support for several programming paradigms. 
Python has a huge collection of comprehensive standard libraries that are expandable. Python's 
key characteristics include its simplicity and ease of learning, freeware and open source status, 
high-level programming language, platform independence, portability, dynamically typed, both 
procedure- and object-oriented design, interpreted, extendable, embedded nature, and sizeable 
library. 

INTRODUCTION TO MACHINE LEARNING

Automatically identifying meaningful patterns in data is a process known as machine learning. In 
the recent years, it has transformed into a common tool for almost any task needing information 
extraction from large data sets. The technology that permeates our lives nowadays includes 
machine learning. Search engines figure out how to provide us the best results while putting 
profitable adverts, anti-spam software figures out how to filter our email communications, and 
fraud-spotting software safeguards credit card transactions. Face recognition is possible with 
digital cameras, while voice recognition is possible with personal assistant apps on smartphones.

INTRODUCTION TO DATA SCIENCE

Data Science is a multidisciplinary field that employs scientific methods, practises, tools, and 
systems to glean knowledge from both structured and unstructured data. Big data, data mining, 
and data analytics are all connected to data science. It is aware of the phenomenon behind the data. 
It uses methods and theories that are derived from a variety of disciplines in the context of 
mathematics, statistics, computer science, and information science.

PYTHON FOR DATA SCIENCE 

The most important data science libraries to be familiar with are as follows: 
• Numpy  
• Matplotlib 
• Scipy 
• Pandas 
• Seaborn 

Numpy: Numpy will greatly improve our ability to manage multi-dimensional arrays. Although 
doing so directly might be challenging, Numpy is the foundation upon which many other libraries 
(indeed, virtually all of them) are built. Simply put, using Pandas, Matplotlib, Scipy, or Scikit
Learn is challenging without Numpy.

Matplotlib: The visualisation of data is crucial. Data visualisation enables us to more effectively 
comprehend the data, locate information that would not be seen in the raw form, and present our 
discoveries to others. Matplotlib is the top-rated and most well-known Python data visualisation 
library. Although it is not user-friendly, it often offers a variety of capabilities, such as bar charts, 
scatterplots, pie charts, and histograms, which are helpful for projecting multidimensional data.  

Scipy: Numerous concepts that are very significant but also complicated and time-consuming are 
covered in mathematics. But Python has a whole scipy library that takes care of this problem for 
us. We will learn how to use this library in this programme, along with a few functions and 
illustrations of how they work. 

Pandas: Pandas is a Python library used for working with data sets. It has functions for analyzing, 
cleaning, exploring, and manipulating data. The name "Pandas" has a reference to both "Panel 
Data", and "Python Data Analysis" and was created by Wes McKinney in 2008 

Seaborn: Seaborn is a Python data visualization library based on matplotlib. It provides a high
level interface for drawing attractive and informative statistical graphics. For a brief introduction 
to the ideas behind the library, you can read the introductory notes or the paper.

SOFTWARE REQUIREMENT 

●  Operating System: Windows 10 
●  Programming Software: Jupyter Notebook 
● Programming Language: Python Programming 

METHODOLGY  

MACHINE LEARNING PROCESS

The data we collected is still raw-data which is very likely to contains mistakes ,missing values 
and corrupt values. Before drawing any conclusions from the data we need to do some data 
preprocessing which involves data wrangling and feature engineering . Data wrangling is the 
process of cleaning and unify the messy and complex data sets for easy access and analysis .Feature 
engineering[7] process attempts to create additional relevant features from existing raw features 
in the data and to increase the predictive power of learing algorithms Our approach to solve the 
problem starts with collecting the raw data need to solve the problem and import the dataset into 
the working environment and do data preprocessing which includes data wrangling and feature 
engineering then explore the data and prepare a model for performing analysis using machine 
learning algorithms and evaluate the model and re-iterate till we get satisfactory model 
performance then compare the results within the algorithm and select a model which gives a more 
accurate results. 
●  Libraries Used: Pandas,Seaborn,MatplotLib,Numpy

What are Bayesian Networks? 
• In general, Bayesian Networks (BNs) is a framework for reasoning under uncertainty 
using probabilities. More formally, a BN is defined as a Directed Acyclic Graph (DAG) 
and a set of Conditional Probability Tables (CPTs). In practice, a problem domain is 
initially modeled as a DAG. 
• Naive Bayes assumes that the variables are independent and comes from a Gaussian 
distribution

THE BAYES THEOREM

  P(A|B) = P(B|A) P(A)/P(B)

• P(A|B) is the posterior probability of class (A, target) given predictor (B, attributes). 
• P(A) is the prior probability of class. 
• P(B|A) is the likelihood which is the probability of predictor given class. 
• P(B) is the prior probability of predictor.

NAIVE BAYES CLASSIFIER ALGORITHM 

The Naive Bayes classifier algorithm is a probabilistic machine learning algorithm that is based 
on Bayes' theorem. It assumes that all features are independent of each other, hence the term 
"naive." Despite this assumption, the Naive Bayes classifier has been proven to perform well in 
many real-world applications, including text classification and spam filtering. 
The steps involved in using the Naive Bayes classifier algorithm for the Titanic survival 
prediction are as follows: 
1. Load the Titanic dataset. 
2. Preprocess the data by handling missing values, encoding categorical variables, and 
scaling numerical features if necessary. 
3. Split the data into training and testing sets. 
4. Train a Naive Bayes classifier on the training data. 
5. Evaluate the performance of the classifier on the testing data using appropriate metrics, 
such as accuracy, precision, recall, or F1 score. 
6. Predict the survival outcome for new, unseen data using the trained classifier.
