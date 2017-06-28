'''
Linear Regression from scratch 
'''

# define a hypothesis in the form: y = mx + b
def hypothesis(theta, x):
    h = theta[0] * x[0] + (theta[1] * x[1])
    return h
