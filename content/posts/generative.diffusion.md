---
title: "Diffusion Models: A brief description"
date: 2026-06-20T23:35:53-04:00
draft: false
math: true
weight: 1
tags: ["machine learning", "sgd", "optimization","generative models"]
---

### Summary  
As argued earlier adding some noise to an otherwise deterministic process can improve sampling and "exploration" 
of the underlying space. Here, we use this intuition to show how a noisy evolution with observed data can help 
learn the constraints that generate this data, and thus enable interpolation to infer new data points from the 
underlying distribution. This writeup is a very brief review of the principles governing diffusion models, and 
is heavily influenced by the review of Lai et.al (2026) which I highly recommend as reading for the interested 
reader (I have provided a citation to it at the end of the writeup).

A central property is that many Markov processes converge to a unique stationary distribution, which often has a 
Boltzmann form. So, if one imagines that there is a random process whose stationary distribution describes the 
set of observed data, it should be possible to "learn" such a random process and thus approximate the data 
distribution. One way to do this is to constrain the process such that the observed data are samples from 
its stationary distribution. In a nutshell this is what generative diffusion models do, and I describe this here. 
Note that there are a couple of points to keep in mind- i) not all Markov processes converge to a stationary 
distribution, indeed SGD does not as discussed earlier and ii) we learn an unnormalized form of the stationary 
distribution (equivalently the "score function"). This latter point is particularly important when modeling physical
systems (like proteins) and will be discussed in more detail at a later time.   

### The Setup  
Let the set of observed data be denoted as $X \equiv \{x_{i}\}_{i = 1}^{N}$, and we assume that they are drawn 
from an unknown probability distribution, $p(x_{i})$ with the data points,$x_i$, being independently and identically 
distributed. We further write $p(x)$ in an exponential form:   

$$  
p(x) = \frac{e^{-\frac{U(x)}{\eta}}}{Z}  
$$  

$Z$ is a normalization constant, the "partition function", such that $\sum_{x}{p(x)} = 1$. It is clear from this 
definition that the exponent is undefined upto a constant value, i.e both $U(x)$ and $U(x) + C$ for an arbitrary 
constant $C$ are potential functions that give the same probability distribution. Furthermore it is known that 
evaluation of the partition function $Z$ is a hard problem, so instead, we attempt to estimate the gradient of this
potential $U(x)$, commonly known as the score function, $s(x) \equiv \nabla U(x) = -\eta\nabla\log p(x)$, directly. 
A physical interpretation is that $U(x)$ is a potential energy that weights the data points $x$. Points with higher 
potential energy are less probable, and vice versa. Thus, regions where datapoints are closely clustered correspond 
to high probabilities, and correspondingly have a lower potential energy etc.   
  
If we knew the score function $s(x)$, we could interpolate between the observed data points and better estimate the 
data distribution, so we aim to approximate it. It is easy to see that the observed distribution is a stationary 
state that satisfies the following condition:  
   
$$  
v(x) \equiv \nabla U(x)  + \eta\nabla\log p(x)  = 0  
$$  
If we interpret $v(x)p(x,t)$ as a current, this is the stationary condition for a Fokker-Planck equation,   
   
$$  
\frac{\partial p(x,t)}{\partial t} = -\nabla\cdot [v(x) p(x,t)]  
$$  
Here, $\eta$ plays the role of a diffusion constant (or equivalently temperature) and the gradient of the potential 
is the drift term. So far, this is pretty straightforward and I have only restated known properties of such equations.   
Unlike with SGD, where we have an estimate of the drift term (the gradient of the mean error there) 
and want to estimate the stationary states of a Fokker-Planck equation, here we have the   
opposite- we have a sampling of the stationary state as represented by the training data, and 
we want to estimate the drift term that results in the observed stationary state.   
   
#### Generative Diffusion Models  
In principle, we could generate data points from the score distribution by propagating a Brownian motion 
with the score as the drift term starting with points from an initial prior of our choice (eg points sampled 
from a Gaussian). However, we have the following conundrum: we have some observations sampled from our unknown 
data distribution but no idea as to the underlying score.   
   
To address this conundrum and estimate the score function given observations, generative diffusion models are 
constructed by doing the following:  

1. Starting with the observed data as initial points, Brownian dynamics is executed with a drift term 
chosen to be proportional to that of a selected prior (eg the Gaussian), and a time dependent diffusion constant.  
2. After these trajectories are propagated for sufficient time (enough that the resulting distribution 
approximates the prior), the same trajectories are time reversed and propagated backward in time to the 
initial start. The drift term of the time reversed trajectories is learnt so as to minimize an error 
function by comparing with the true score when both are evaluated on the training data and generated 
trajectories.  
   
The second step is possible because of a key mathematical property that the drift term governing the time 
reversed trajectories includes the true score function. More precisely if the forward drift term is 
$ f(x,t)$, with the forward Ito process given by:  
   
$$  
dx(t) = f(x,t)dt + g(t)dw_{t}  
$$  

Then the corresponding time-reversed Ito process is: (Hasegawa 1976, Anderson 1982)  

$$  
d\tilde{x} = [f(\tilde{x},t) - g^{2}(t)\nabla\log p_{t}(\tilde{x})]dt +g(t)d\tilde{w}_t  
$$

This result was derived by Hasegawa in 1976 who constructed time-reversed dynamics corresponding to a given 
forward time evolution due to a linear dissipative operator, and separately by Anderson in 1982 who derived 
the time reversed Ito process corresponding to a given forward process. 
  
The appearance of the score function, $s(x) \equiv \nabla\log p_{t}(x)$ in the time reversed dynamics is the 
key property that enables the construction of generative diffusion models.   
  
The remaining ingredient needed to specify the training of a generative diffusion model is to specify the 
error function to be optimized. The most obvious objective function would be to perform score matching:   
   
$$  
L_{sm}[p_t]= \sum_{t}\Big\{\int{dx_{t} p_{t}(x_t)\vert s_{\phi}(x_{t},t) - \nabla_{x}\log p_{t}(x_{t})\vert^{2}}\Big\}  
$$  
The score function $s_{\phi}(x,t)$ is represented by a deep neural network with hyper parameters $\phi$. 
This objective function is intractable as the function  $\log p_{t}(x)$ is unknown.  
  
To render the training of these models tractable, Vincent (2011) developed a denoising method. 
Specifically, consider an auxiliary variable $y$, with a known conditional probability distribution, 
$p_{\sigma}(y\vert x)$ and scale $\sigma$ (eg if the conditional probability is Gaussian, $\sigma$ is its
variance). Then

$$  
p_{\sigma}(y) = \int{dx p_{\sigma}(y\vert x)p_{data}(x)}  
$$  
 
We attempt to minimize the error $L_{sm}[p_{\sigma}]$ with the score over the auxiliary variable as the 
target. In the limit $\sigma \rightarrow 0$, $L_{sm}[p_{\sigma}] \rightarrow L_{sm}[p_{data}]$. Although,
this error function is also intractable, it can be shown that estimating the error w.r.t the conditional 
distribution, $p_{\sigma}(y\vert x)$, which is tractable, is equivalent to minimizing $L_{sm}$ ! We briefly
demonstrate this below:

First, consider the following loss function:

$$
L_{d}[p_{\sigma}(y\vert x)] \equiv \sum_{t}{\int{dy dx p_{\sigma}(y\vert x)p_{data}(x)\Big\vert s_{\phi}(y,t) - \nabla_{x_{t}}\log p(y\vert x)\Big\vert^{2}}}
$$

This is equivalent to minimizing for every $(y,t)$
$$
L' = \int{dx p_{\sigma}(x\vert y)\Big\vert s_{\phi}(y,t) - \nabla_{y}\log p_{\sigma}(y\vert x)\vert^{2}}
$$   

We get that $L'$ is minimized when
$$
s^{*}_{\phi}(y) = \int{dx p_{\sigma}(x\vert y)\frac{\nabla p(y\vert x)}{p_{\sigma}(y\vert x)}}
$$

This reduces to 
$$
s^{*}_{\phi}(y) = \nabla_{y}\log p_{\sigma}(y)
$$

At time $t$, substituting $y = x_{t}$ and $p(y\vert x) = p_{t}(x_{t}\vert x)$ gives 
$$
s^{*}(x_{t},t) = \nabla\log p_{t}(x_{t})
$$

Thus, the true score $s^{*}(x_{t},t) = \nabla\log p_{t}(x_{t})$ is the value at which the 
$\textit{numerically tractable}$ denoising loss is minimized:

$$
\nabla_{x_{t}}\log p_{t}(x_{t}) = \arg\min L_{sm}[p_{t}]
$$
where
$$
L_{sm}[p_{t}] \equiv \sum_{t}{dx_{t}dx p_{t}(x_{t}\vert x)p_{data}(x)\lVert s_{\phi}(x_{t},t) - \nabla_{x_{t}}\log p_{t}(x_{t}\vert x)\rVert^{2}}
$$

The time reversal properties of a Brownian process coupled with the "denoising" loss function that is numerically
tractable together provide the basic pieces required to train a generative diffusion model, where stochastic 
trajectories are run forward in time from the given training data samples, and the loss function minimized. Once
the score function is estimated, trajectories are run backward in time until $t = 0$ starting from sampling of the 
prior distribution at $t = T$ to obtain new samples. These can be either deterministic, governed by the probability
flow equation:

$$
x_{0} = x_{T} + \int_{t=T}^{t = 0}{d\tau \big[f(x_{\tau},\tau) - \frac{g^{2}(\tau)}{2}s^{*}_{\phi}(x_{\tau},\tau)\big]}
$$

or the stochastic time evolution update:
$$
x_{t -\delta t} = x_{t} - \big[f(x_{t},t) - g^{2}(t)s^{*}_{\phi}(x_{t},t)\big]\delta t + g(t)\sqrt{\delta t}\xi
$$

$\xi \sim N({\bf 0},{\bf I})$ is a Gaussian random variable, $\delta t$ is the timestep.

#### Remarks
The construction of generative diffusion models is possible because the time reversed evolution retains
information regarding the score distribution function, and is made computationally tractable due to the denoising 
approach first pioneered by Vincent (2011). The time reversal property discussed here is general, and reflects a 
general property of dissipative (linear) time evolution. 

The time reversal properties point to some future avenues of investigation. A clearer understanding of the effects 
of a "round trip" evolution of the initial data distribution, to an arbitrary prior and back could aid in estimation 
of not only the score, but free energies of the data distribution, or more correctly free energy differences between 
substates in the data distribution. This could be of value to problems of protein structure prediction. A second, 
related avenue could be to exploit these time reversal properties to understand the stochastic thermodynamics of 
generative diffusion models. I hope to write about some of these avenues if I find anything of interest in future 
notes. 

#### References

These are by no means exhaustive, and I have included the ones most directly relevant to the writeup above. The
first reference contains references to much of the relevant literature. 

1. C. -H. Lai, Y. Song, D. Kim, Y. Mitsufuji and S. Ermon, The Principles of Diffusion Models, 
arxiv:2510.21890 (2026) https://arxiv.org/abs/2510.21890v2 . I found this paper to be a comprehensive and very 
clear review of diffusion models and the principle they are based on. Much of what I have explained in this note
is derived and heavily borrowed from this review.
2. H. Hasegawa, On the Construction of a Time Reversed Markoff Process. Progress of Theoretical Physics, 1976 
vol 55(1), pgs. 90-105. doi:https://doi.org/10.1143/PTP.55.90 . This paper derives a comprehensive and elegant
treatment of time reversal dynamics of general Markovian processes whose time evolution is governed by a linear 
dissipative operator.
3. B. D. O. Anderson, Reverse-time diffusion equation models, Stochastic Processes and their Applications, 1982
I haven't actually read this paper, but have seen proofs of the theorem proved in it. I found the paper a bit 
tricky to read and understand. 
4. P. Vincent, A Connection Between Score Matching and Denoising Autoencoders. Neural Computation, 
vol 23, no. 7, pp. 1661-1674, 2011, doi: 10.1162/NECO_a_00142. This paper introduced the denoising procedure this
makes diffusion models computationally tractable. 