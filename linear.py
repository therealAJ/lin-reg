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

def cfd(theta, X, y, sub_i, m):
    sumErr = 0
    for i in xrange(m):
        x = X[i]
        h_i = hypothesis(theta, x)
        y_i = y[i]
        err = (h_i - y_i) * x[sub_i]
        sumErr += err
    cost = (1 / m) * sumErr
    return cost

def gradient_descent(X, y, theta, m, alpha):
    opt_theta = []
    constant = alpha * (1 / m)
    for i in xrange(len(theta)):
        cost = cfd(theta, X, y, i, m)
        updated_theta = theta[i] - constant * cost
        opt_theta.extend(updated_theta)
    return opt_theta

def Linear_Regression(X, Y, alpha, theta, iter):
    m = len(X)
    for i in xrange(iter):
        opt_theta = gradient_descent(X, y, theta, m, alpha)
        theta = opt_theta
    return opt_theta