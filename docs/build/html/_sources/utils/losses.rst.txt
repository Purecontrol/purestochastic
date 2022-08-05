################
    Losses
################

Implementation of new loss functions, especially stochastic loss functions. The package is also compatible with 
all the loss functions from `tensorflow <https://www.tensorflow.org/api_docs/python/tf/keras/losses>`_ as well 
as custom loss functions. There are two main ways to use loss functions in tensorflow : 

    1. Use the loss function directly with `tf.keras API <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_ : 

        >>> model.compile(
        >>> loss=gaussian_negative_log_likelihood,
        >>> ....
        >>> )

    2. Use the loss function as a standalone function : 

        >>> y_true = [[5], [10], [35]]
        >>> pred = [[[6, 1]], [[7, 1]], [[36, 0.8]]]
        >>> gaussian_negative_log_likelihood(y_true, pred).numpy()
        2.756748

Here is the list of the new losses :

.. contents:: :local:
    :depth: 2

Gaussian Negative Log Likelihood
-----------------------------------

.. autofunction:: purestochastic.common.losses.gaussian_negative_log_likelihood

