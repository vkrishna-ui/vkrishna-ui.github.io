---
title: "Information Geometry and SGD"
date: 2026-06-02T23:35:53-04:00
draft: false
math: true
weight: 2
---
### Summary
Here I describe a modification to SGD, which results in the dynamics converging to a stationary distribution over
parameter space. It turns out that this modified SGD method is equivalent to a Brownian dynamics in a curved space,
with the distance (metric) of that curved space given by the covariance or Fisher information matrix of the loss 
gradient.

### Natural gradients in stochastic gradient descent
The diffusion matrix, $C(\mathbf{\theta})$ that falls out of the probabilistic treatment of SGD in my previous note 
is positive definite, as it is a covariance matrix. To recapitulate, $C(\mathbf{\theta})$ is 
defined as (I have dropped the coefficient related to minibatch size):
$$
C(\theta) \equiv \left\langle{\bf f}(s,\mathbf{\theta})\odot{\bf f}(s,\mathbf{\theta})
\right\rangle - {\bf f}(\mathbf{\theta})\odot{\bf f}(\mathbf{\theta}). 
$$ 
As before, the force for a sample $s$ is ${\bf f}(s,\mathbf{\theta}) = -\nabla_{\theta}E(s,\mathbf{\theta})$ 
is the negative gradient of the sample error. Importantly, if we interpret the error $E(s,\mathbf{\theta})$ 
as a form of energy, the force is a score function for a Boltzmann distribution induced by this 
energy, and $C(\mathbf{\theta})$ becomes the Fisher Information matrix for this distribution.

To understand the underlying geometry, an inner product can be defined as follows:
$$
\langle{\bf v},{\bf u}\rangle_{c} \equiv {\bf v}\cdot C(\theta)\cdot{\bf u}
$$
and correspondingly, the norm of a vector, $\lVert v\rvert^{2} \equiv \langle{\bf v},{\bf v}\rangle_{c}$, then it 
can be shown that this norm satisfies the Cauchy-Schwarz inequality (this is because the covariance is symmetric and
positive definite:
$$
\lVert {\bf v} + {\bf u}\rVert^{2} \leq (\lVert {\bf v}\rVert + \lVert {\bf u}\rVert)^2
$$

Now, consider the following infinitesimal changes for a given sample $s$:
$$
d\xi(s,\vec{\theta}) \equiv ({\bf f}(s,\theta) - {\bf f}(\theta))\cdot d\vec{\theta}
$$

This is simply the displacement along the direction of the loss gradient for that sample, compared
to the mean displacement for an infinitesimal change in parameters. Then, the (Euclidean) distance 
traversed upon the infinitesimal change in parameters when averaged over all samples $s$ is:
$$
\left\langle\lVert d\xi\rVert^{2}\right\rangle = d\vec{\theta}\cdot C(\vec{\theta})\cdot d\vec{\theta}
$$

Using the definition of the inner product, we get a nice relation:
$$
\left\langle\lVert d\xi\rVert^{2}\right\rangle = \lVert d\vec{\theta}\rVert_{c}^{2}
$$
(I have used the subscript c, to distinguish our inner product norm from an Euclidean norm). This has a 
neat geometric interpretation as saying that the average mean square change in error upon changing
parameters of our ML model infinitesimally, is equal to the infinitesimal distance in our (curved)
parameter space,endowed by a metric given by the covariance. This essentially says that a curved 
metric space (at least for some metrics) is an average over randomly fluctuation flat geometries, 
which is a nice way of thinking about curved geometries, although I am not sure how general this 
is, and whether it holds for arbitrary metrics.
In this curved space, we can rewrite the Fokker-Planck equation also as an inner product:
$$
\partial_{t}\sigma = \left\langle\nabla_{\theta}, \Bigg[-C^{-1}\sigma{\bf f} + \frac{1}{2\vert B\vert}\nabla_{\theta}\sigma \Bigg]\right\rangle_{c}
$$
In a curved space with a metric $C(\theta)$, Amari defined a "natural" gradient to be:
$$
\nabla_{C}\sigma(\vec{\theta}) \equiv C^{-1}\cdot\nabla_{\vec{\theta}}\sigma(\vec{\theta})
$$
This definition identifies the true gradient, or direction of steepest descent in a curved space. 
Thus, for our example a true Fokker-Planck equation in inner product space will be of the form:

$$
\partial_{t}\sigma = \left\langle\nabla^{*}_{C},\Bigg[-C^{-1}f\sigma + \frac{1}{2\vert B\vert}\nabla_{C}\sigma\Bigg]\right\rangle_{c}
$$

Here, $ \nabla^{*}_{C} $ is the adjoint operator for the natural gradient. This is equivalent to:
$$
\partial_{t}\sigma  = -\nabla^{*}_{C}\cdot{\bf J}\sigma
$$ 
With a current given by: 

$$
{\bf J} = {\bf f}(\vec{\theta}) - \frac{1}{2\vert B\vert}\nabla_{\vec{\theta}}\ln\sigma
$$

This expression for the current is independent of the covariance, and thus this dynamics
converges to a steady state Boltzmann distribution, when the current ${\bf J} = 0 $ ! 
The resulting steady state Boltzmann distribution is:

$$
\rho_{ss}(\vec{\theta}) = \frac{1}{Z}e^{-2\vert B\vert E(\vec{\theta})}
$$

The corresponding stochastic natural gradient dynamics, with minibatch $B$ has the form:
$$
\vec{\theta}_{t + 1} = \vec{\theta}_{t} - \eta(t) C^{-1}(\vec{\theta})\cdot\nabla_{\theta}E(\vec{\theta}_{t},B)
$$ 
This dynamics is guaranteed to result in a steady state probability over the space of parameters, 
which also is the desired true parameter probability distribution obtained by minimizing the 
error! The tradeoff here is that the estimation of the gradient becomes computationally harder, 
as it involves estimating the inverse of a Fisher Information matrix. 

NB: Please feel free to inform me of any errors or oversights in the derivations presented here. 

#### Further Reading
S.I Amari Neural Computation (1998) 10(2) 251-276; doi: https://doi.org/10.1162/089976698300017746
Natural gradients were introduced by Amari into machine learning and are an entry into the field
of Information geometry.
