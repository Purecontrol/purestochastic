import tensorflow as tf
from tensorflow import keras
from keras.regularizers import Regularizer
from keras.utils.generic_utils import get_custom_objects

class Orthonormality(Regularizer):
    r"""
    A regularizer that applies a Orthonormality penalty [1]_ to 2D and 3D kernel.
    The penalty is computed as : 

    .. math::
         \text{loss} = \text{lambda_coeff} \frac{1}{n} \sum_{i=1}^{K} \sum_{j=1}^{K} ((C^TC)_{i,j} - I_{i,j})

    with C, the kernel matrix. If C is a 3D tensor the operation is the mean over the new dimension.

    The penalty forces the kernel to be an orthonormal matrix.

    Arguments
    ---------
    miso : boolean (default: True)
        Tell the regularizer if the kernel is a 2D or 3D matrix.
    lambda_coeff : float (default : 1)
        The regularization factor.

    References
    -----------
    .. [1] Tagasovska, Natasa and Lopez-Paz, David. « Single-model uncertainties for deep learning ». 
        In : Advances in Neural Information Processing Systems 2019.Nips (2019), p. 1-12. issn : 10495258. 
        arXiv : 1811.00908.

    """

    def __init__(self, miso=True, lambda_coeff=1):  

        self.lambda_coeff = lambda_coeff
        self.miso = miso

    def __call__(self, x):

        if self.miso:
            K = x.shape[1]
            return self.lambda_coeff * tf.reduce_mean(tf.square(tf.transpose(x) @ x - tf.eye(K)))
        else:
            K = x.shape[2]
            return self.lambda_coeff * tf.reduce_mean(tf.square(tf.einsum('ijk, ijp -> jkp', x, x) - tf.eye(K)))

    def get_config(self):

        return {'miso' : self.miso, 'lambda_coeff': float(self.lambda_coeff)}

get_custom_objects().update({'orthonormality': Orthonormality})