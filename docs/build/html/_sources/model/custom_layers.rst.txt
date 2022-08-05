Custom Layers
################



:class:`Dense` layer in `Tensorflow <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`_ uses a 2D kernel 
of shape ``(n_input, n_output)``. This module adds new Dense layers with 2D, 3D and 4D kernel.


The names of the new Dense layers are defined as ``Dense<Input_dim>to<Output_dim>`` with the batch size dimension included. 
For example, the layer :class:`Dense2Dto3D` takes as input a 2D tensor of shape ``(batch_size, n_input)`` and outputs a tensor of
shape ``(batch_size, units_dim1, units_dim2)``.


The new layers can be used as usual tensorflow layers. They are useful when the model outputs parameters of a 
distribution. For instance, if the model predicts the mean and the variance of a gaussian distribution for 4 variables, 
it is interesting to have an output shape equal to ``(batch_size, 4, 2)``. It is then possible using this piece
of code : 

>>> inputs = Input(shape=(input_dim,))
>>> x = Dense(100, activation="relu")(inputs)
>>> outputs = Dense2Dto3D(4, 2, activation=MeanVarianceActivation)(x)
>>> model = Model(inputs=inputs, outputs=outputs)
>>> model.output_shape
(None, 4, 2)

Here is the list of the new layers :

.. contents:: :local:
    :depth: 2

A detailed presentation of each layer is available below along with an image describing the operations performed 
by each layer.

Dense2Dto3D
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: purestochastic.model.layers.Dense2Dto3D
    :no-undoc-members:

The following figure represents the linear operation performed by the layer :class:`Dense2Dto3D`. 
If ``activation`` is specified, the activation function is applied to the output of the linear
operation described below.

.. image:: Dense2Dto3D.drawio.svg
    :width: 500
    :align: center

Dense3Dto3D
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: purestochastic.model.layers.Dense3Dto3D
    :no-undoc-members:

Dense3Dto2D
~~~~~~~~~~~~~~~~~~~~~~


.. autoclass:: purestochastic.model.layers.Dense3Dto2D
    :no-undoc-members:

Dense3Dto4D
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: purestochastic.model.layers.Dense3Dto4D
    :no-undoc-members: