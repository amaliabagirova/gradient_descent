# Gradient descent
This is programming assignment in which I implemented gradient descent and weighted linear regression from scratch.

## Problem 1. Gradient descent
**Task 1** 

Considering a function $$f(x) = x^2 - 15\sin \left(\tfrac{\pi}{3}x\right).$$ 

Implementing funtions `f(x)` and `grad_f(x)`, which evaluate function and its gradient in any given point `x`. 


I start with a base class for the gradient descent algorithm. Recall that gradient descent can be used to find a local minimum of a function $f(x)$

The algorithm is the following:

* Choose initial point $x_0$
* Make steps in the direction of the anti-gradient
$$x_{t+1} = x_t - \nu \nabla f(x_t), \\$$
* Repeat until stopping criterion is satisfied. Possible stopping criterions are:
    - $\|x_{t+1} - x_t\| < \varepsilon$
    - $\|\nabla f(x_{t+1})\| < \varepsilon$
    
**Task 2** 

Implementing the following methods in the class below:
* `step`, which makes one step of the gradient descent algorithm 
* `stoppin_criterion`, I will use the first option, i.e. $\|x_{t+1} - x_t\| < \varepsilon$
* `find_min`, which finds local minimum of a functions

**Task 3** 

Assume that the starting point is fixed $x_0 = 9$. I will tune hyperparameters of the method, so that it coverges to a **global** minimum (or to the point wich is close enough to the global minimum). Value of the target function in the obtained point should smaller that `-12.5`. I will tune different parameters:
- learning rate (`lr`)
- maximal number of iterations (`max_iter`)
- tolerance of the stopping criterion (`eps`)

