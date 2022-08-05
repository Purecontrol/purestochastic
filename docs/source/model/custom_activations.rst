########################
Custom Activations
########################

Implementation of new stochastic ``activations``. At present, there is only one
activation functions but it will be easy in the future to add more possibility. 

There are often used with the last layer of a network. For example, the :class:`MeanVarianceActivation`
is especially used with the :class:`Dense2Dto3D` or :class:`Dense3Dto4D` layers : 

>>> inputs = Input(shape=(input_dim,))
>>> x = Dense2Dto3D(100, activation="relu")(inputs)
>>> x = Dense3Dto3D(100, activation="relu")(x)
>>> outputs = Dense3Dto4D(4, 2, activation=MeanVarianceActivation)(x)
>>> model = Model(inputs=inputs, outputs=outputs)

Here is the list of the new activations :

.. contents:: :local:
    :depth: 2

.. TIP::
    In order to specify the parameters of a custom activation, you can use the following syntaxes :
    
        * ``model.add(Activation(lambda x: MeanVarianceActivation(x, activation="relu")))``
    
        * ``Dense3Dto4D(4, 2, activation=lambda x: MeanVarianceActivation(x, activation="relu"))``


MeanVarianceActivation
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: purestochastic.model.activations.MeanVarianceActivation

