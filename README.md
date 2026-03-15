# Logistic Regression

This project implements logistic regression in python without using any additional libraries (yet)

The goal is to understand how logstic regression works internally, including gradient descent, sigmoid activation and parameter updates.

## Model
The logistic regression model computes

z = w * x + b

The output is passed through the sigmoid function:

p = 1 / (1 + e^-z)

This converts the result into a probability between 0 and 1.

## Training Process

The model is trained using gradient descent.

Steps:

1. Initialize weight and bias
2. Compute predictions using the sigmoid function
3. Calculate prediction error
4. Compute gradients
5. Update the parameters

The parameters are updated using:

w = w - learning_rate * dw  
b = b - learning_rate * db

## Purpose

This project was created to understand how logistic regression works internally rather than relying on machine learning libraries.
