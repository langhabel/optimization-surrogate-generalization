# Learning to Optimize

When faced with a new task, humans usually do not start from scratch but incorporate knowledge obtained from previously solved problems. We implement the same procedure for optimization problems using deep learning: Our neural network learns and generalizes over similar problems!

## Table of Contents

1. [Introduction](#introduction) <br/>
   1.1 [An Intuitive Explanation](#intuitive-explanation) <br/>
   1.2 [A Scientific Explanation](#scientific-explanation)<br/>
2. [User Interface](#user-Interface)  <br/>
3. [Modules](#modules)  <br/>
   3.1 [Uncertainty Measurement](#uncertainty-measurement)<br/>
   3.2 [Balancing Exploration and Exploitation](#balancing-exploration-and-exploitation)<br/>
   3.3 [Function Approximation Using Deep Learning](#function-approximation-using-deep-learning)<br/>
4. [NIPS](#nips)  <br/>

<a name="introduction"/><br/>
## 1 Introduction

<a name="intuitive-explanation"/> <br/>
### 1.1 An Intuitive Explanation

- **Problem**: We focus on optimization problems of arbitrary dimensionality that can be represented with functions.<br/>
   *Example*: There are similar but different propellers for airplanes, hovercrafts, and ships. Their performance depends on features, e.g. width and length of different components. We want to maximize performance given similar scenarios.

- **Input**: Some samples of the underlying function (performance over features) are available and serve as input for the neural network.

- **Output**: The neural network proposes a new place for sampling.<br/>
   *Example*: Suppose you have much experience (samples) for building *airplane* propellers. Now you want to build a *hovercraft* propeller, but you don't know where to start! The neural network proposes a prototype that will most likely perform decently. Once you have tested this first prototype, the new knowledge can be incorporated in the model. Thus, the neural network learns along the way and improves its suggestions.

- **Performance**: Increased efficiency by more than 50% [<sup>[1](#myfootnote1)</sup>]

<a name="scientific-explanation"/><br/>
### 1.2 A Scientific Explanation
Optimization constitutes a highly relevant but complex task in many industries. Special focus lies on efficiency, as obtaining samples from the objective functions is typically particularly expensive. We address this problem by using neural networks as surrogate models and estimates of uncertainty in the surrogates to orchestrate the optimization process. We evaluate multiple methods for uncertainty estimation and apply them to balance between exploration and exploitation. The approach is economical regarding the number of required samples. We further propose a method to build a meta-model over a class of similar optimization problems, which we show more than halves the number of samples required in new problems by generalizing from old ones.

<a name="user-Interface"/><br/>
## 2 User Interface

You can find a notebook-demo for the generalization module [here](https://github.com/langhabel/optimization-surrogate-generalization/blob/master/src/demo.ipynb). The following commands are all you need. It is that easy.

**Instantiation**:
- `Generalizer(dim, initial_samples_X, initial_samples_Y, grid_size)`
   - `dim`: Space of data, array with shape (2,dimensions), rows refer to min/max
   - `initial_samples_X`, `initial_samples_Y`: Samples, array with shape (data points,dimensions), prior scaling of domain necessary (e.g. using `utils.scale_01(x, original_range)`)
   - `grid_size`: Amount of evaluated points along each dimension (integer) [<sup>[2](#myfootnote2)</sup>]

**Proposition of new sampling location**:
- `generalizer.get_next_sampling_location(slice)`
   - `slice`: *One* optimization problem, e.g. *airplane propeller optimization* within *propeller optimization*. The entire model consists of many (n-1)-dimensional *slices*, e.g. functions for propellers of hovercrafts, ships and planes.
   - `utils.scale_restore(data)` transforms data back to original space (reverses prior scaling)
   
**Incorporation of new sample**:
- `generalizer.incorporate_new_sample(x_new, y_new)`
   - `x_new`, `y_new`: *x* refers to features (e.g. propeller-width), *y* to respective labels (e.g. performance)

Exemplary convergence criteria can be found in the demo-notebook.

<a name="modules"/><br/>
## 3 Modules

<a name="uncertainty-measurement"/><br/>
### 3.1 Uncertainty Measurement

This module measures prediction-uncertainty:

- `uncertainty.compute_uncertainty(train_X, train_Y, test_X, dims, method='Ensembles')`
   - `train_X`, `train_Y`: Training data and labels
   - `test_X`: test-grid (Where should uncertainty be computed?)
   - `dims`: Dimensionality
   - `method`: Method for obtaining uncertainty, either `Kriging`, `XDist`, `Ensembles` or `MCDropout` [<sup>[3](#myfootnote3)</sup>]

<a name="balancing-exploration-and-exploitation"/><br/>
### 3.2 Balancing Exploration and Exploitation

This module calculates *expected improvement* for each future sampling location. The method balances exploration and exploitation based on uncertainty estimates [<sup>[4](#myfootnote4)</sup>]. This assures efficient sampling.

- `expected_improvement.compute_expected_improvement_forrester(prediction, uncertainty, y_min, grid)`
   - `prediction`: Predicted function for slice (array with values for each grid-point)
   - `grid`: Represents space (array with points to evaluate)
   - `uncertainty`: Function displaying prediction-uncertainty (array with values for each grid-point)
   - `y_min`: Minimum value of all samples in slice

<a name="function-approximation-using-deep-learning"/><br/>
### 3.3 Function Approximation Using Deep Learning 

Given sample points, this module approximates continuous functions with arbitrary dimensionality. [<sup>[5](#myfootnote5)</sup>]

<a name="nips"/><br/>
## 4 NIPS
An [extended abstract](http://bayesiandeeplearning.org/2016/papers/BDL_9.pdf) has been presented at a [NIPS 2016 workshop](http://bayesiandeeplearning.org/2016/index.html). A copy can also be found in [/docs/](https://github.com/langhabel/optimization-surrogate-generalization/tree/master/docs).

## 5 Details

- [<a name="myfootnote1">1</a>]: [Graphical illustration](https://image.ibb.co/gp8XPR/Selection_006.png): Efficiency is measured by the required amount of samples for reaching the global minimum. We used the [*six hump camel function*](https://www.sfu.ca/~ssurjano/camel6.html) to test our implementation.

- [<a name="myfootnote2">2</a>] Scaling to higher dimensionality exponentially increases the search space. We advise using a sparse grid in those cases.

- [<a name="myfootnote3">3</a>]: [Performance overview for uncertainty-measurement methods](https://image.ibb.co/hgNZ4R/Selection_005.png)
   - [`Kriging`](https://en.wikipedia.org/wiki/Kriging): *Gaussian process regression*, interpolated values are modeled by a Gaussian process using variance as uncertainty estimates.<br/>
   *Advantage*: Efficient calculation; *disadvantage*: Independent of neural network.
   - `XDist`: Increasing Euclidean distance to samples leads to increased uncertainty.<br/>
   *Advantage*: Easy; *disadvantages*: too peaky, uneven results, independent of neural network
   - [`Ensembles`](http://ieeexplore.ieee.org/abstract/document/1216214/): Using several neural networks with different random weight initializations. Mean over outputs constitutes prediction, variance depicts uncertainty.<br/>
   *Advantage*: Uncertainty estimates depend on neural network; *disadvantage*: computationally expensive<br/>
   *Future work*: Ensemble-uncertainty-estimates could be used as training data for a new neural network, which could then exclusively be used on new data. This would increase computational efficiency.
   - [`MCDropout`](http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_3d801aa532c1ce.html): *Monte-Carlo dropout*: Approximation of deep Gaussian process by application of dropout (multiple times) in test instead of training. Mean of output constitutes prediction, variance depicts uncertainty.<br/>
   *Advantage*: Uncertainty estimate depends on neural network<br/>
   *Disadvantage*: Tendency that higher curvature (in the function) is modelled by scarce number of neurons. Thus, dropout disproportionally affects those regions. However, high curvature can be present in already explored regions, leading to poor uncertainty estimates in our scenario.

- [<a name="myfootnote4">4</a>]: Based on two-stage approach according to [formula (81)](https://image.ibb.co/bD3yW6/Selection_004.png), Section 4.3.1, Recent advances in surrogate-based optimization, Forrester and Keane
   - **[Graphical illustration](https://image.ibb.co/gWvW16/Selection_003.png)**: A Gaussian distribution displays the probability of possible prediction values for each grid-point *x*. Predictions decreasing the current slice-minimum are most relevant. Thus, the formula calculates the mean of the "area enclosed by the Gaussian distribution below the best observed value so far".
   - **Intuition**: The *expected improvement* evaluates all unobserved points *x*. If the *expected improvement* is high for point *x*, then sampling at this point will probably result in a new slice-minimum.


- [<a name="myfootnote5">5</a>]: We target zero-bias regression as samples are assumed to be noise-free. Implementation in [tensorflow](https://www.tensorflow.org/) with following parameters:
   - **Hidden Layer**: 2
   - **Hidden units**: 200/layer (*More* units -> local minima; *less units* -> underfit)
   - **Activation**: *ReLu* (1st layer) + *Sigmoid* (2nd layer)
      - *ReLu*: Sharp inflexions and higher latitude; *sigmoid*: Smoothness (higher underfit)
   - **Optimizer**: [Adam](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) (*initial learning rate*=0.01, higher rate -> coarseness)
   - **Training epochs**: 2000 (*More*: Oscillation between samples; *fewer*: underfit)
   - **Weight/bias initialization**: Truncated normal distribution (scaled up (*stddev*=2) to achieve desired overfit)
   - **Regularization** None as it leads to underfit (*dropout* and *L2-norm weight decay regularization* are implemented, but not activated)
   - **Scaling**: Domain has to be scaled to [0,1] for excellent results

## 6 License
Copyright 2016 Jonas Langhabel and Jannik Wolff. See [License](https://github.com/langhabel/optimization-surrogate-generalization/blob/master/LICENSE) for details.


