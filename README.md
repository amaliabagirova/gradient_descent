# Gradient descent
This is programming assignment in which I implemented gradient descent and weighted linear regression.

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

**Task 4** 

Let's now make the learning rate dependent on the step number.  

$$
x_{t+1} = x_t - \nu_t \nabla f(x_t), \\
\nu_t = \frac{\nu_0}{t}
$$

Where $\nu_0$ is the initial value and it is reduced with a constant speed. 

---
## Problem 2. Weighted Linear Regression 
---

### Linear Regression Recap

In simple linear regression we have assumed that all the observations are of the same 'importance' to the model. In practice it is not always the case. Due to different reasons, it may happen that some observations are more valuable for us than others. 

Let us start with recapping the simple linear regression.

* **Model** with k features:
$$
a(x) = w_0 + w_1x^1 + \dots w_kx^k = \langle w, x \rangle,\\
x = (1, x^1, \dots, x^k)
$$

* **Dataset:** 
$$\text{design matrix } X \in \mathbb{R}^{N \times k+1},\\
\text{target values }y \in \mathbb{R}^{N}$$

* **MSE Loss**:
$$
L = \tfrac{1}{N}\| y - Xw\|^2_2
$$

We obtained the matrix form of the MSE loss inthe following manner:


$$
L(a, X) = \frac{1}{N}\sum_{i=1}^N (y_i -  a(x_i))^2 =  \frac{1}{N}\sum_{i=1}^N (y_i -  \langle w, x_i \rangle)^2 =  \frac{1}{N}(y - Xw)^T(y - Xw) =  \frac{1}{N} \| y - Xw\|^2_2
$$


In this case optimal parameters (that minimize the loss function) can be written in a closed-form:

$$
\hat{w} = \left(X^T X\right)^{-1}X^Ty
$$

### Weighted Linear Regression
Assume now, that some observations in our dataset are more "important" that others. E.g. we know that for some points the measurements are less accurate and want to reduce the weight of such observation. Another possible reason: we assume more recent observations to be more relevant and want to account for that in our loss function. 

* The **model**  (exactly the same):
$$
a(x) = w_0 + w_1x^1 + \dots w_kx^k
$$

* The **dataset**: 
$$\text{design matrix } X \in \mathbb{R}^{N \times k+1},\\
\text{target values }y \in \mathbb{R}^{N}$$

In addition, we have vector of weights, which reflects the importance of each observation:
$$v = (v_1, \dots, v_N)$$ 

* **MSE loss**:

We will change loss function, so that it includes the weights:
$$
L(a, X) = \frac{1}{N}\sum_{i=1}^N v_i(y_i -  a(x_i))^2
$$

In the matrix form, it will looks like this:
$$
L(a, X) =  \tfrac{1}{N}\| V^{1/2}(y - Xw)\|^2_2
$$

Where $V$ is matrix with weight $v$ on the diagonal and zeros elsewhere:

$$
V = \begin{pmatrix}
v_1 & 0 & \cdots & 0\\
0  & v_2 & \cdots & 0 \\
\vdots & \vdots & \ddots& \vdots \\
0 & 0 & \cdots & v_n
\end{pmatrix}
$$

In this task we will train weighted linear regression using both closed form solution and gradient descent algorithm

--
**Task 2.1** [1 pt]  <a class="anchor" id="task2_1"></a>

Calculate gradient of the weighted MSE loss with respect to parameters for the model

$$
\nabla_w L =  \nabla_w \tfrac{1}{N}\| V^{1/2}(y - Xw)\|^2 = ?
$$

**Hints:** You can use formulas from the lecture. 

Given vector $x \in \mathbb{R}^n$ and matrix $A \in \mathbb{R}^{k \times n}$
$$
\nabla_x \| x\|^2_2 = 2x
$$

$$
\nabla_x Ax = A^T
$$

Using the formula frm the gradient, implement the function `weighted_mse_grad`, which calculates gradient for any given vector $w$.  

To find analytical solution we need to solve
$$
\nabla_w L = 0
$$
---
**Task 2.2** [1 pt] <a class="anchor" id="task2_2"></a>
Write a function `weighted_lr`, which calculates optimal parameters for the weighted linear regression

**Hint**. Module `linalg` from `scipy` package has many usefull operations, including matrix inversion. It is highly likely that you will need it to solve this task. For example `sp.linalg.inv(A)` will return inverse of the matrix `A`.

---
**Task 2.3** [1 pt] <a class="anchor" id="task2_3"></a>
Using weigth matrices `v_none` and `v_smart`, find optimal parameters of the weighted linear regressions. Call them `w_none` and `w_smart` correspondingly. 
