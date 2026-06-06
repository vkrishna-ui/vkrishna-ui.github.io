---
title: "Stochastic Gradient Descent"
date: 2026-05-26T23:35:53-04:00
draft: false
math: true
---
### SGD as an early diffusion model
Generative diffusion models aim to reconstruct an underlying probability distribution from a given 
set of samples/observations (this is a bit crude in description, as diffusion models actually 
approximate a score function, partly because estimating partition functions is hard). They 
have been very successful in methods that predict protein structures from a given sequence. 
Interestingly enough, stochastic gradient descent (SGD) that is widely used to train deep learning models 
has ideas in common to those used in diffusion models, and this commonality is explored here. 

#### A basic description of SGD
A basic recipe in training a ML model is to estimate and minimize a mean loss function that measures the 
difference between model predictions and the true positives over a training set. A common method is 
\\$\textit{gradient descent}\\$ to iterate the mean loss by moving along its gradient w.r.t to the model 
parameters until a minimum is reached. This approach can be computationally expensive when the number 
of training samples is very large, as is often the case with modern deep learning methods. 
Robbins and Monro (Annals of Statistics 1951) first introduced SGD, wherein instead of evaluating the 
gradient for a loss estimated over the entire training set, one estimates gradients for a randomly 
chosen subset of training samples.

In mathematical terms, let $\theta$ be a set of parameters of a ML model that we want to optimize, 
by minimizing an expected loss $\left\langle E(X,\theta)\right\rangle$ with the expectation taken over the distribution of 
(training) samples. In traditional gradient descent we update the parameters using the rule:

$$ 
\theta(t+1) = \theta(t) − \eta_{t}\nabla_{\theta}\left\langle E(X,\theta)\right\rangle
$$
Here the expectation value of the loss is approximated by the empirical mean loss over the set of training samples. In SGD, the expectation value is replaced by a mean over a randomly chosen subset, $B$, of samples, or even a single sample. Ie. 

$$
\theta(t+1) = \theta(t) - \frac{1}{\vert B \vert}\eta_{t}\sum_{s \in B}{\nabla_{\theta}E(x_{s},\theta)}
$$

This modification reduces computational cost, and we would like to understand the (stochastic) dynamics described in the equation above. If we define  $\eta_{t}{\bf f}(B,\theta) \equiv \eta_{t}\nabla_{\theta}E(B,\theta)$, we can define a conditional probability:

$$
\begin{equation}
K(\vec{\theta}_{t+1}\mid\vec{\theta}_{t};\eta_{t}) = 
\sum_{\bf f}{\delta\left(\vec{\theta}_{t+1} - \vec{\theta}_{t} - 
\eta_{t}{\bf f}(B,\theta_{t})\right)P( {\bf f} \vert \vec{\theta}_{t} )}.
\end{equation}
$$ 
We have introduced a probability $P({\bf f}\vert \theta_t)$ due to the property that ${\bf f}(B,\theta_{t})$ is a random variable because the subset of samples $B$ are chosen randomly. To develop a dynamics, we use a Fourier representation:
$$
		P({\bf f}\vert\theta_{t}) = \sum_{B}\delta({\bf f} - {\bf f}(B,\theta_{t}))Q(B)
$$
with $Q(B) \equiv \prod_{s \in B}{q(s)}$ being the distribution of training samples $s$. On using a Fourier representation for the Dirac delta function:
$$
P({\bf f}\vert \theta_{t}) = \sum_{B}{\int{d{\bf k} Q(B) \exp (i{\bf k}\cdot[{\bf f} - {\bf f}(B,\theta_{t}])}}
$$
We can exchange the integral and summation signs to obtain:
$$
P({\bf f}\vert \theta_t) = \int{d{\bf k}e^{i{\bf k}\cdot{\bf f} - \Gamma({\bf k},\theta_{t})}}
$$




### Block Equations
To show standard standalone mathematical proofs, break them into a multi-line equation display block:

