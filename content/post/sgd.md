---
title: "Stochastic Gradient Descent"
date: 2026-05-26T23:35:53-04:00
draft: false
math: true
weight: 1
tags: ["machine learning", "sgd", "optimization"]
---
### SGD as an early diffusion model
Generative diffusion models aim to reconstruct an underlying probability distribution from a given 
set of samples/observations (this is a bit crude in description, as diffusion models actually 
approximate a score function). They have been very successful in methods that predict protein 
structures from a given sequence. Interestingly enough, stochastic gradient descent (SGD) that 
is widely used to train deep learning models has ideas in common to those used in diffusion 
models, and this commonality is explored here. I show that SGD induces the evolution of a 
probability distribution governed by a Fokker-Planck equation, with a diffusion matrix 
proportional to a covariance matrix of gradients; and its score function given by the 
gradient of the loss function employed in training the machine learning model. 


#### What is described here? 
In simple terms,this note examines a common method used to train the parameters of a 
machine learning model by minimizing the error between the predictions of the model and the 
observed ground truth. Intuitively, to find the "best" parameters that minimize an error, 
we vary the parameters and update them in the direction in which the error reduces the 
most; this is known as gradient descent. However, this requires looking at every single 
piece of training data before taking each step. For the massive datasets used to train 
modern AI models, this is very slow and computationally expensive. 

To speed this up, a Stochastic Gradient Descent (SGD) method is used. Instead of looking at 
all the data at once, a random subset (a "minibatch") is chosen to guess which way is downhill.
Doing this also introduces noise into the error estimation. This note shows that this noise 
converts the gradient descent trajectory into a random, diffusive walk. Furthermore the
analysis shows that the presence of randomness improves exploration of parameter space by 
preventing the model from being trapped into a suboptimal portion of the parameter space, thus
improving the estimation of the model's parameters to make for a better performing model. 

#### A basic description of SGD
A basic recipe in training a ML model is to estimate and minimize a mean loss function that 
measures the difference between model predictions and the true positives over a training set. 
A common method is $\textit{gradient descent}$ where the mean loss is reduced by moving along 
its gradient w.r.t to the model parameters until a minimum is reached. This approach can be 
computationally expensive when the number of training samples is very large, as is often the 
case with modern deep learning methods. Robbins and Monro (1951) first introduced SGD, wherein 
instead of evaluating the gradient for a loss estimated over the entire training set, one 
estimates gradients for a randomly chosen subset of training samples.

In mathematical terms, let $\theta$ be a set of parameters of a ML model that we want to 
optimize, by minimizing an expected loss $\left\langle E(X,\theta)\right\rangle$ with the 
expectation taken over the distribution of (training) samples. In traditional gradient descent 
we update the parameters using the rule:

$$ 
\theta(t+1) = \theta(t) − \eta_{t}\nabla_{\theta}\left\langle E(X,\theta)\right\rangle
$$
Here the expectation value of the loss is approximated by the empirical mean loss over the set 
of training samples. In SGD, the expectation value is replaced by a mean over a randomly chosen 
subset, $B$, of samples, $s$:
$$
\theta(t+1) = \theta(t) - \frac{1}{\vert B \vert}\eta_{t}\sum_{s \in B}{\nabla_{\theta}E(s,\theta)}
$$

This modification reduces computational cost, and we analyse the (stochastic) dynamics described 
in the equation above. If we define  ${\bf f}(B,\theta) \equiv -\nabla_{\theta}E(B,\theta)$, we 
can define a conditional probability:

$$
\begin{equation}
K(\vec{\theta}_{t+1}\mid\vec{\theta}_{t};\eta_{t}) = 
\sum_{B,{\bf f}}{\delta\left(\vec{\theta}_{t+1} - \vec{\theta}_{t} + 
\eta_{t}{\bf f})\right)P[{\bf f} = {\bf f}(B,\theta_t)]}.
\end{equation}
$$ 
Note that this removes the $\delta$ function from the sum over $B$ To estimate 
$P[{\bf f}] = \sum_{B}{P[{\bf f} = {\bf f}(B,\theta_{t})]}$ we use a Fourier representation:
$$
		P({\bf f}) = \sum_{B}\delta({\bf f} - {\bf f}(B,\theta_{t}))Q(B)
$$
with $Q(B) \equiv \prod_{s \in B}{q(s)}$ being the distribution of training samples $s$. On using a Fourier representation for the Dirac delta function:
$$
P({\bf f}) = \sum_{B}{\int{d{\bf k} Q(B) \exp (i{\bf k}\cdot[{\bf f} - {\bf f}(B,\theta_{t})])}}
$$
We can exchange the integral and summation signs to obtain:
$$
P({\bf f}\vert \theta_t) = A\int{d{\bf k}e^{\vert B\vert \big(i{\bf k}\cdot{\bf f} - \Gamma({\bf k},\theta_{t})\big)}}
$$

The function $\Gamma$ is a cumulant generating function and so using a steepest descent 
approximation, we can derive the probability to be Gaussian, with a mean given by the force 
averaged over samples $s$, 
${\bf f}(\theta) \equiv -\nabla_{\theta}\left\langle E(s,\theta)\right\rangle$, and a variance 
given by the variance matrix of the force, i.e. 
$$
C(\theta) \equiv \left\langle{\bf f}(s,\theta)\odot{\bf f}(s,\theta)
\right\rangle - {\bf f}(\theta)\odot{\bf f}(\theta). 
$$ 
The product $\odot$ is the outer product of vectors.Note that there is an additional 
prefactor $\frac{1}{\vert B\vert}$ that goes into the actual covariance matrix that 
enters the equations, but we separate it out from the definition and instead include 
it in the equations. Thus the use of minibatches is equivalent to adding a white noise 
component to the regular gradient dynamics. Here is a widget to see the difference between 
a simple gradient descent and the brownian dynamics induced by SGD, in 1-dimension. 

{{< optimization_sim >}}

With this Gaussian distribution in hand, stochastic gradient descent can be shown to sample a probability distribution, 
$\rho(\theta,t)$ that satisfies a Fokker-Planck equation:

$$
\boxed{
\frac{\partial\rho}{\partial t} = -\nabla\cdot{\bf f}(\theta)\rho + \frac{1}{2\vert B\vert}\nabla\cdot C(\theta)\cdot\nabla\rho
}
$$ 

Due to the dependence of $C(\theta)$ on $\frac{1}{\vert B\vert}$, it is clear that as the batch 
size grows to be very large, the dynamics begins to resemble that of a simple gradient descent. 
Thus SGD results in a diffusion process on the loss landscape, and has a stationary solution 
when the probability current is zero:
$$
{\bf f}(\theta) = \frac{1}{\vert B\vert}C(\theta)\cdot\nabla\log \rho
$$
Since the mean force ${\bf f}(\theta)$ is itself the gradient of a scalar potential, 
${\bf f}(\theta) = -\nabla\left\langle E(S,\theta)\right\rangle$, with the expectation value 
taken w.r.t the true (but unknown) data distribution, it is easy to see that in general SGD 
does not converge to a stationary state, but instead results in limit cycles. On the other hand, 
the classical gradient descent method can result in quasi-stationary states where the force 
becomes zero, suggesting that the limiting behavior of the two methods is quite different. 
The behavior of SGD is thus strongly influenced by mini-batch size, where a large size, 
$\vert B\vert$ relative to the largest eigenvalue of the covariance matrix $C(\theta)$ makes 
SGD a small perturbation of gradient descent, while smaller batch sizes can induce sufficiently 
large fluctuations to allow for a different sampling of the loss landscape. 

This illustrates a possible reason why SGD works better than gradient descent, beyond it being 
computationally cheaper. Choosing relatively small minibatches to calculate the gradient injects 
noise into the sampling of hyperparameters, allowing for a better exploration of the loss 
landscape, while gradient descent does not necessarily allow for a similar exploration over the 
same amount of training time, as it drives the parameters towards the nearest local critical 
point on the loss landscape. 

An interesting implication is that the addition of some noise to a system's 
dynamics often improves sampling and generalization outcomes, by facilitating greater 
sampling and exploration. This insight is also at the heart of modern generative diffusion 
models. As a matter of fact, generative diffusion models also rely on generating dynamics 
according to a Fokker-Planck equation similar in structure to what is derived here. 

##### A concluding note
A natural question to ask is- why is it useful to understand the evolution of a probability
distribution over parameters? After all, in the simplest terms, one is only interested in an
optimal (or "good enough") set of parameter values that make a machine learning model a good
predictor, rather than probability distributions over parameters. Classical gradient descent
in fact does provide such an optimal value of parameters at the end of training. However,
the introduction of randomness through minibatch selection "widens" this choice, and instead
results in a distribution of parameters, with the optimal set being a mode of this probability
density function. 


Notably, as the minibatch size, $\vert B\vert$ increases, the associated
probability density function becomes increasingly concentrated around its (highest) modal value,
and the parameter values begin to "anneal" towards this value. In reality given the sparsity
of true minima in a high dimensional loss landscape, due to the finite time taken for training, 
gradient descent would likely result in parameter values associated with a critical point, that 
may not be a true minimum, while by adding noise to the dynamics SGD possibly enables a better
sampling of the loss landscape, resulting in a greater likelihood of a parameter set that enables
better generalization. This is, of course, a lot of handwaving, that is further complicated by
the fact that SGD results in limit cycles, rather than a true stationary distribution. 


However, as I discuss in a subsequent writeup, replacing simple gradients by a "natural" gradient does
result in the dynamics converging to a Boltzmann distribution over the loss function in parameter
space, with an effective temperature that is inversely proportional to the minibatch size, 
$\vert B\vert$. Thus, here taking the limit of infinite minibatch size is equivalent to a 
simulated annealing process where the temperature is taken to zero, and results in the optimal
parameter set being a true minimum of the loss landscape. 

##### Some Further Reading (This is by no means exhaustive)
That SGD induces a Fokker-Planck equation has been derived, and the properties of this 
dynamics are a fairly active area of research. I reference a couple of papers that could
serve as an entry point. 
 
1. P. Chaudhari and A. Soatto arXiv:1709.11029 (2017); doi: https://doi.org/10.48550/arXiv.1710.11029
   This paper discusses the properties of limit cycles that SGD converges to, and also examines how the
   dynamics varies with learning rate. It also has some nice empirical studies of the eigenvalue properties 
   of the diffusion matrix (the covariance matrix), and connections with gradient flows.
   
2. S. Mandt, M. D. Hoffman and D. M. Blei, JMLR (2017) 18,1-35; doi:https://doi.org/10.48550/arXiv.1704.04289
   This is an interesting treatment of SGD as a form of Bayesian Inference. 
   
2. H. Robbins and S. Monro, Ann. Math. Statist. (1951) 22(3): 400-407 ; doi:https://doi.org/10.1214/aoms/1177729586
   This is the original paper that introduced the stochastic gradient descent method.


