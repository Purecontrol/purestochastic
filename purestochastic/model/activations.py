import tensorflow as tf
from tensorflow import keras
from keras import activations
from keras.utils.generic_utils import get_custom_objects


def MeanVarianceActivation(x, activation=None):
    r"""An activation function used for the mean and variance pair. 

    This activation function supposes that the last dimension of x is of length 2 with 
    the first part for the ``mean`` and the second part for the ``variance`` :
    
        * :math:`\hat{\mu} = x[ : , \cdots , :  , 0]`

        * :math:`\hat{\sigma}^2 = x[ : , \cdots , : , 1]`
    
    It applies the activation given in argument for the mean and the exponential activation function
    for the variance so that it's positive. 
    
    It's especially used when it's necessary to output the mean and the variance of a gaussian distribution 
    at the end of the network like in the Mean Variance Estimation Method [1]_ [2]_ . 

    Parameters
    ----------
    x : ndarray or Tensor
        Input tensors.
    activation : func or str, optional
        Activation function to use. If you don't specify anything, 
        no activation is applied (ie. "linear" activation: `a(x) = x`).
    
    Returns
    -------
    A new tensor to which the activation function has been applied. 


    References
    ----------
    .. [1] Abbas Khosravi et al. « Comprehensive Review of Neural Network-Based Prediction Intervals
        and New Advances ». In : IEEE Transactions on Neural Networks 22.9 (2011), p. 1341-1356.
        doi : 10.1109/TNN.2011.2162110.

    .. [2] D.A. Nix et A.S. Weigend. « Estimating the mean and variance of the target probability
        distribution ». In : Proceedings of 1994 IEEE International Conference on Neural Networks
        (ICNN94). T. 1. 1994, 55-60 vol.1. doi : 10.1109/ICNN.1994.374138.
    """

    # Get activation for the mean
    activation = activations.get(activation)
    
    # Split the vector into mean and variance elements
    mean, variance = tf.split(x, 2, axis=-1)

    # Apply different activation for the mean and the variance
    if activation is not None:
        mean = activation(mean)
    variance = activations.exponential(variance)

    # Concatenate the variance and the mean to recover the original shape
    x = tf.concat([mean,variance], axis=-1)

    return x

get_custom_objects().update({'MeanVarianceActivation': MeanVarianceActivation})