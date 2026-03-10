# What is logistic rgression
Logistic regression is a binary classification algorithm,
meaning it tries to answer a yes or no type of question.

In my example, x is the hours studied, y is the actual result

So my example model is trying to learn "Given the number of hours studied, what is the probability that the student passes"

The parameters : weight and bias
weight --> w = 0.0
bias --> b = 0.0

The weight controls how strongly the input affect the prediction
- if w is positive, bigger x gives a bigger output
- if w is negative, bigger x gives a smaller output

In this example we expect w to become positive because more studying should increase the probability of passing

The bias shifts the prediction left or right
It helps the model fit the data better even when the input is 0
It acts as an offset 

## Learning Rate
learning rate = 0.1
 
This control how big each update is.
- too big -->  model may overshoot and become unstable
- too small - model learns very slowly

Th learning rate is basically how aggressively we change the weight and bias each step

## Epochs 
epochs = 1000
An epoch means one full pass through the entire dataset

In this example I have 6 training examples.
So in 1 epoch the models looks at the 6 examples all at once during that epoch

If a model trains for 1000 epochs, it repeats that full process 1000 times
The reason for that is that the model doesn't magically know the right weight bias at the start. It improves with every iteration

So:
- 1 epoch = one full training cycle over the data
- More epochs = chances to improve the parameters

## Sigmoid Function 
def sigmoid(z):
return 1/(1 + m.exp(-z))

Before the function is called, the model computes the linear equation.
z can be any real number, but for binary classification we want a probability between 1 and 0.

That's why we apply the sigmoid function. It squashes the numbers into the range of 0 and 1
- If z is really negative the probability is near 0
- If z is very positive the probability is near 1

## Training Loop 
It repeats the training 1000 times.

Inside each epoch, the model
1. Makes predictions
2. Measures error
3. Calculates gradients
4. Updates weights and bias

That's its learning process

## Step 1
At the start of each epoch:
- dw will store the gradient for the weight 
- db will store the gradient for the bias
- n is the number of data points

You reset them because each epoch calculates fresh average gradients from the whole dataset

## Step 2
```for i in range(n):```

This goes through each data point one by one.

## Step 3
Forward Pass

It means that we use the current w and b to make a prediction

What happens:
For each input
1. compute the linear output
2. convert it into a probability

So y_pred is the model's predicted probability of passing.

## Step 4 
Calculating the error contribution
``` (y_pred - y[i]) ```

This is a prediction error.
- if the prediction is bigger than the actual value the error is positive
- if the prediction is smaller than the actual value the error is negative 

It helps the model understand which way to move

## Step 5
Gradient for the weight.
``` dw += (y_pred - y[i]) * x[i] ```
dw --> tells us how mch the weight should change
We multiply it by x[i] because the weight affect the prediction through the inout
A larger input contributes more strongly to the weight's effect

So it's basically telling the model : 
" Take the error, and scale it by how much this input influenced the prediction "

## Step 6
``` db += ( y_pred -y[i]) ```
Bias is not multiplied by x[i] because is just added directly
It doesn't depend on the input value.
So for the bias gradient we only need the error term.

## Why is updating the weight and bias needed?

Because at the start, w = 0 and b = 0 so the model is basically clueless.
It predicts badly.

The gradients tell us:
- how wrong the model is 
- whether to increase or decrease w 
- whether to increase or decrease b

then we update them: 
w -= lr * dw
b -= lr * db

This is called gradient descent
But why do we subtract?

Because gradients point to the direction of increasing error
We want to reduce the error so we move to the opposite direction.

Why average the gradients?
dw /= n
db /= n
You sum the gradient contributin from all training examples, the divide by n.
That gives the average gradient.
We do that because otherwise the update size would depend heavily on how many data point you have. 
Averaging makes training more stable and consistent.


Final prediction stage:
After training , you print the learned parameters.

Then you move to testing: 
``` for x in x: 
prob = sigmoid(w * x +b)
prediction = 1 if prob >= 0.5 else 0```

