import tensorflow as tf
import numpy as np

@tf.function
def gaussian_negative_log_likelihood(y, prediction):
    r"""Gaussian negative log likelihood loss function.

    The loss function computes the gaussian negative log likelihood between the ground truth values 
    :math:`y=(y_1, \ldots, y_n)` and the predictions which are the mean and the variance of a gaussian
    distribution : 
    
        * :math:`\hat{\mu} = \text{prediction}[ : , \cdots , :  , 0]`

        * :math:`\hat{\sigma}^2 = \text{prediction}[ : , \cdots , : , 1]`
    
    Mathematically, the loss function is defined as : 

    .. math::
        \mathcal{L}(y, \hat{\mu}, \hat{\sigma}^2) = \frac{1}{2n} \displaystyle\sum_{i=1}^{n} \Bigg[\ln(\hat{\sigma}^2(x_i,\theta))+\frac{(y_i-\hat{\mu}(x_i,\theta))^2}{\hat{\sigma}^2(x_i,\theta)} \Bigg] + \frac{1}{2}\log(2 \pi)
    
    If ``y`` is not a 1D array, the batch dimension needs to be the first and the output value is
    the mean over all other dimensions.


    Parameters
    ----------
    y : tf.Tensor or np.ndarray
        Ground truth values.
    prediction : tf.Tensor or np.ndarray
        The predicted values.

    Returns
    -------
    tf.float32 
        Value of the Gaussian negative log likelihood.

    Input shape
    -----------
        (N+2)-D tensor with shape: ``[batch_size, d_0, ...,  d_N]``.

    Output shape
    ------------
        (N+3)-D tensor with shape: ``[batch_size, d_0, ..., d_N, 2]``.

    Note
    ----
    The likelihood is a product of density functions and so the values can be between :math:`[0, \infty]`.
    The negative log likelihood can thus be negative if the likelihood is greater than 1. Therefore, don't
    worry if your loss function is negative, it's often the case.
    
    """

    # Separate the mean and the variance
    mean, var = tf.unstack(prediction, axis=-1)

    # Compute the Gaussian Negative Log Likelihood
    log_var = tf.math.log(var)
    squared_error = tf.math.pow(tf.math.subtract(mean,y),2)
    return tf.math.reduce_mean(log_var + tf.math.divide(squared_error,var))/2 + (1/2)*tf.math.log(2*np.pi)

