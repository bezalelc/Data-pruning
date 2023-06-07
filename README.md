# Data-pruning
Research how different methods of dataset pruning can cause better performance and computation saving in machine learning

Academic supervisor: Doctor Yehuda Hassin

# Abstract
This project is a research project with Dr. Yehuda Hassin about deep learning, the
purpose of the project is to test whether and how it is possible to prune part of existing
dataset and obtain comparable results on deep learning models in order to save
resources (time, computing power, data collection) and/or achieve better performance
of the model error.
There are researches that have already been done on the subject with various
suggestions and metrics on how to evaluate the quality and difficulty of the data, find
the most significant data for training and prune the rest at the very beginning of the
learning process of the model, within the project we implemented them and offered
additional metrics to evaluate the quality and importance of each data training
example according to information that we have on the dataset already at the beginning
of learning and we divided the data according to the level of difficulty that each
metric gave us.
In addition, we implemented a self-unsupervised metric KM that knows how to
classify the dataset according to the level of difficulty without receiving labels and
additional information about the dataset, so KM metric can also be useful in problems
where we try to learn data without labels with the help of self-supervised models.
To test how each part of the data affects the learning process of the model, we
performed a series of experiments that tested how each part of the data affects the
model's error and we found that if there is a large enough data then the model will
learn better from the more difficult examples and we can give up on the least
important part of the dataset and on the other hand if the dataset is small then the
model also need the easy examples.
Another common problem with datasets is that in order to prepare the data for
training, we need to hire people to look at the data and label it, and because it is a
large amount of data, there are sometimes mistakes in the labeling that can confuse
the model and damage its correctness. In this project we found that with the help of
EL2N and pred sum metrics it is possible to find the "noisy" examples and remove
them from the data.

# Files
The folder "code" includes all the code of the project's experiments in the notebooks, in each notebook it is indicated at the beginning which experiments it includes. In addition, there are several shared code files between the notebooks:
 - train.py
 - utils.py 
 - train_vegetable_dataset.py
 - utils_vegetable_dataset.py

# Dataset
In this research we tested 3 datasets: 
 - CIFAR10
 - CIFAR100
 - vegetable dataset from here: https://www.kaggle.com/datasets/ahmadalqawasmeh/vegetables-images
