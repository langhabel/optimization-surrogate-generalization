# Optimization Surrogate Generalization

## Abstract

Optimization constitutes a highly relevant but complex task in many industries. Special focus lies on e ciency, as obtaining samples from the objective functions is typically particularly expensive. We address this problem by using neural networks as surrogate models and estimates of uncertainty in the surrogates to orchestrate the optimization process. We evaluate multiple methods for uncertainty estimation and apply them to balance between exploration and exploitation. The approach is economical regarding the number of required samples. We further propose a method to build a meta-model over a class of similar optimization problems, which we show more than halves the number of samples required in new problems by generalizing from old ones.

## Modules

All of the following can be replaced according to your own preferences. The interfaces are described below; more details can be taken from the respective code documentation. The heart of the contribution lies in the generalization module.

### Uncertainty Measuring

This module implements a collection of different methods to obtain uncertainty in the sample search space. Call:

`uncertainty.compute_uncertainty(train_X, train_Y, test_X, dims, method='Ensembles')`

Where `train_X` and `train_Y` are the training data and labels, `test_X` the test data grid, `dims` ist the number of problem space dimensions and `method` is the method used to obtain uncertainty: either Kriging, XDist, Ensembles or MCDropout.

### Exploitation-Exploration Weighing

Collecting samples by evaluating the underlying function lies at the heart of optimizing a surrogate. This is assumed to be very expensive, so we implemented the two-stage approach according to formula (81), Section 4.3.1, Recent advances in surrogate-based optimization, Forrester and Keane, to efficiently balance exploitation and exploration based on uncertainty estimates.

`expected_improvement.compute_expected_improvement_forrester(prediction, uncertainty, y_min, grid)`

`prediction` is the predicted fit to the function based upon the same samples, given in the space of 'grid', `uncertainty` the respective uncertainty obtained from above function at all gridpoints based upon the samples, `y_min` the minimum function value of the samples in the slice and `grid` the grid used to represent the space.

### Generalization

Main program which optimizes n dimensional search spaces. The Generalizer optimizes faster by maintaining a n+1 dimensional surrogate model which incorporates the samples of already solved similar problems with shared features. The additional axis is not vital to solve each optimization problem but connects similar problems from the same class. We thus speak from generalizing between optimization problems.

You can find an easy to use and adapt notebook demo for the generalization module [here](https://github.com/langhabel/optimization-surrogate-generalization/blob/master/src/demo.ipynb). The basic usage relies on the following two commands to add new samples in the model and then have it compute the next preferrable sampling location in the given current slice:

`generalizer.incorporate_new_sample(x_new, y_new)`

`generalizer.get_next_sampling_location(slice)`

## License

Copyright 2016 Jonas Langhabel and Jannik Wolff. See [License](https://github.com/langhabel/optimization-surrogate-generalization/blob/master/LICENSE) for details.
