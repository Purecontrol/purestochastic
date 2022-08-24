###################
    Regularizers
###################

Implementation of new regularizers. The package is also compatible with  all the regularizer functions from 
`tensorflow <https://www.tensorflow.org/api_docs/python/tf/keras/regularizers>`_ as well as custom regularizer 
functions. There are two main ways to use loss functions in tensorflow : 

    1. Use the string identifiant : 

        >>> dense = tf.keras.layers.Dense(3, kernel_regularizer='orthonormality')

    2. Use the class to specify parameters

        >>> dense = tf.keras.layers.Dense(3, kernel_regularizer=Orthonormality(lambda_coeff=0.01))

Here is the list of the new regularizers :

.. contents:: :local:
    :depth: 2

Orthonormality
-----------------------------------

.. autofunction:: purestochastic.common.regularizers.Orthonormality