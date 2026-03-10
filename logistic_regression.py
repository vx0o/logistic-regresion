import math as m

#data set
x = [1, 2, 3, 4, 5, 6]
y = [ 0, 0, 0 , 1, 1, 1]



w = 0.0 # weight
b = 0.0 #bias 
learning_rte = 0.1
epochs = 1000

# sigmoid function

def sigmoid(z): #z is the linear combination of inputs and weights plus bias
    return 1 / (1 + m.exp(-z)) 
#sigmoid function maps any real-valued number into the (0, 1) interval
# making it suitable for binary classification problems like logistic regression.

#training loop
for epoch in range(epochs):
    dw = 0 
    db = 0
    n = len(x) # number of training examples
    
    for _ in range(n):
        
        #forward pass 
        z = w * x[_] + b
        y_pred = sigmoid(z) #turns the linear output into probaility
        
        #gradients 
        #the gradients measure how wrong the model is and in which direction the parameters should 
        #move to reduce the error.
        dw += (y_pred - y[_]) * x[_] # the error term (y_pred - y[_]) is multiplied by the input x[_]
        #to calculate the contribution of that input to the gradient of the weight
        
        
        db += (y_pred - y[_]) # the error term (y_pred - y[_]) 
        #the error term (y_pred - y[_]) is used directly to calculate the gradient of the bias,
        # as the bias does not depend on any input features.
        
    #avergage gradients
    dw /= n
    db /= n
    
    
    #update weights and bias, gradient descent step
    w -= learning_rte * dw
    b -= learning_rte * db
    
print(f"Trained weight: {w}, bias: {b}")

for x in x:
    prob = sigmoid(w * x + b)
    print(f"Input: {x}, Predicted Probability: {prob:.4f}")
    
    prediction = 1 if prob >= 0.5 else 0
    print(f"Hours studies: {x}, Predicted Probability of passning: {prob:.4f}, Predicted Class: {prediction}")
    