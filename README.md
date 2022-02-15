# Image Classification with Convolutoional Neural Network

## 1.    Introduction

Image classification is where a computer analyse an image and identify the ‘class’ the image falls under. (Or a probability of the image being part of a ‘class’.) A class is essentially a label.. 

Deep learning is a subset of artificial intelligence (AI) a type of machine learning. It involves the use of computer systems known as neural networks. In neural networks, the input filters through hidden layers of nodes. These nodes each process the input and communicate their results to the next layer of nodes. This repeats until it reaches an output layer, and the machine provides its answer. 

Image classification with deep learning most often involves convolutional neural networks, or CNNs. In CNNs, the nodes in the hidden layers don’t always share their output with every node in the next layer (known as convolutional layers).

In this porjectn an Image processing model is developed using CNN.

## 2. Application

As Autonomous vehicles are increasing to be an eco-friendly alternative for conventional transport, the need for high accuracy functioning of self-driving vehicles is vital. One factor that makes self-driving vehicles safer is their ability to analyze the traffic signs and interpret them well.As Autonomous vehicles are increasing to be an eco-friendly alternative for conventional transport, the need for high accuracy functioning of self-driving vehicles is vital. One factor that makes self-driving vehicles safer is their ability to analyze the traffic signs and interpret them well. 

## 3. Dataset

The dataset is obtained from [gtsrb-german-traffic-sign](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
The dataset we are working on contains more than50000 images of different traffic signs categorized into 43 classes. 

## 4. Steps

- The relevant libraries are imported. 
- The Dataset is loaded and explored.
- The Dateset is split into train and test.
- Build a CNN model.
    - The architecture of our CNN model
    - Conv2D layer –  we will add 2 convolutional layers of 32 filters, size of 5*5, and activation as relu
    - Max Pooling – MaxPool2D with 2*2 layers
    - Dropout with a rate of 0.25.0
    - 2 Convolutional layer of 64 filters and size of 3*3
    - Dropout with a rate of 0.25
    - Flattenn layer to squeeze the layers into 1 dimension
    - Dense, feed-forward neural network(256 nodes, activation=”relu”)
    - Dropout Layer(0.5)
    - Dense layer(nodes=46, activation=”softmax”)
    - MaxPool2D – Maximum pooling layer is used to reduce the size of images
    - Dropout – Dropout is a regularization technique to reduce overfitting
    - Flatten – to convert the parrel layers to squeeze the layers
    - Dense –  for feed-forward neural network
    - the last layer will have an activation function as softmax for Multi-class classification.
- Train and validate the model.
    - Our model received an accuracy of 95% on training data.
- Test the Model 
    - The dataset contains a test folder that has different test images and a test.csv file that contains details related to the image path and respective labels. Again we will load the data using pandas and resize it to the shape of 30*30 pixels and convert it to a NumPy array. After processing test images we will check the accuracy of the model against actual labels.
-  Save the Model for future use as well.