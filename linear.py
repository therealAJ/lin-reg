'''
Linear Regression from scratch 
Uppercase params represent matrices
Lowercase params represent vectors

Thanks to Andrew Ng's Stanford Coursera notes for providing
equations
'''

# define a hypothesis in the form: y = mx + b
def hypothesis(theta, x):
    h = theta[0] * x[0] + (theta[1] * x[1])
    return h

def cost_function(theta, X, y, m):
    sse = 0
    for i in xrange(m):
        x = X[i]
        h_i = hypothesis(theta,x)
        y_i = y[i]
        err = (h_i - y_i)
        sqErr = err**2
        sse += sqErr
    const = 1 / (2*m)
    cost = const * sse
    return cost


