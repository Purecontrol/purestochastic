import tensorflow as tf
from tensorflow import keras
from keras import initializers, regularizers, constraints, activations, backend
from keras.layers import InputSpec
from keras.utils.generic_utils import get_custom_objects

class Dense2Dto3D(tf.keras.layers.Layer):
    r"""An adaptation of the densely-connected NN layer that outputs a 3D tensor from a 2D tensor.

    :class:`Dense2Dto3D` is a change of the :class:`Dense` layer when the kernel is a tensor of order 3.
    It implements the dot product between the inputs and the kernel along the last axis of the inputs 
    and axis 0 of the kernel : 
    
    >>> output = activation(tensordot(input, kernel) + bias). 
    
    It's like having a :class:`Dense` layer with ``units_dim1*units_dim2`` units followed by a :class:`Reshape` 
    layer with a target shape of ``(units_dim1, units_dim2)``.

    Parameters
    ----------
    units_dim1 : int
        Dimensionality of the first dimension of the output space.

    units_dim2 : int
        Dimensionality of the second dimension of the output space.

    activation : func or str, default: None
        Activation function to use. If you don't specify anything, 
        no activation is applied (ie. "linear" activation: `a(x) = x`).

    use_bias : boolean, default:True
        Indicates whether the layer uses a bias matrix.

    kernel_initializer : str or dict or func, default:'glorot_uniform'
        Initializer for the `kernel` weights tensor.

    bias_initializer : str or dict or func, default:'zeros'
        Initializer for the bias matrix.

    kernel_regularizer : str or dict or func, optional
        Regularizer function applied to the `kernel` weights tensor.

    bias_regularizer : str or dict or func, optional
        Regularizer function applied to the bias matrix.

    activity_regularizer : str or dict or func, optional
        Regularizer function applied to the output of the layer (its "activation").

    kernel_constraint : str or dict or func, optional
        Constraint function applied to the `kernel` weights tensor.

    bias_constraint : str or dict or func, optional
        Constraint function applied to the bias matrix.

    Input shape
    -----------
        2D tensor with shape: ``(batch_size, input_dim)``.

    Output shape
    ------------
        3D tensor with shape: ``(batch_size, units_dim1, units_dim2)``.
    
    """

    def __init__(self, 
                 units_dim1,
                 units_dim2,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        # Superconstructor of the class tf.keras.layers.Layer
        super(Dense2Dto3D, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs)

        # Specify the dimension of the layer
        self.units_dim1 = int(units_dim1) if not isinstance(units_dim1, int) else units_dim1
        self.units_dim2 = int(units_dim2) if not isinstance(units_dim2, int) else units_dim2
        if self.units_dim1 < 0 or self.units_dim2 < 0:
            raise ValueError(f'Received an invalid value for `units_dim1` or `units_dim2`'
            f', expected positive integers. Received: units_dim1={units_dim1}, units_dim2=={units_dim2}')

        # Preprocess activation, bias, initialiers, regularizers, constraints
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Specify input shape
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        """
        Creates and initializes the kernel weight tensor and the bias 
        matrix of the layer. The rank of the input_shape needs to be 2.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the input.
        """

        # Check if the dtype is compatible
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('A Dense layer can only be built with'
                        f' a floating-point dtype. Received: dtype={dtype}')

        # Check if the input shape is compatible
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to a Dense layer '
                        'should be defined. Found None. '
                        f'Full input shape received: {input_shape}')
        self.input_spec = InputSpec(ndim=2, axes={-1: last_dim})

        # Initialize kernel weights
        self.kernel = self.add_weight(
                        name='kernel',
                        shape=[last_dim,  self.units_dim1, self.units_dim2],
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        dtype=self.dtype,
                        trainable=True)

        # Initialize bias weights
        if self.use_bias:
            self.bias = self.add_weight(
                            name='bias',
                            shape=[self.units_dim1, self.units_dim2],
                            initializer=self.bias_initializer,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                            dtype=self.dtype,
                            trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        r"""
        Defines the dot product between the kernel and the input. 
        If `self.use_bias=True`, the output is summed with the bias matrix.
        If `self.activation` is not None, an activation function is applied 
        element-wise on the output. 

        Parameters
        ----------
        inputs : shape=[batch_size, input_dim]
            Input tensor.

        Returns
        -------
        outputs : shape=[batch_size, units_dim1, units_dim2]
            Output tensor.
        """

        # Change input dtype if incompatible with layer
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        # Tensor contraction between the inputs and the kernel on the axis i : 
        # outputs_{b,o,k} = \sum_{i} inputs_{b,i} kernel_{i,o,k} 
        # The abreviations are : 
        # b : batch
        # i : last_dim of inputs
        # o : self.units_dim1
        # k : self.units_dim2
        # The shape of the input is (num_batches, last_dim) and the shape
        # output is (num_batches, self.units_dim1, self.units_dim2)
        outputs = tf.einsum('bi,iok->bok', inputs, self.kernel)

        # Add bias. The shape of outputs is (num_batches, self.units_dim1, self.units_dim2)
        # and of self.bias is (self.units_dim1, self.units_dim2). Consequently, broadcasting
        # is used whithin the sum to output a tensor of shape 
        # (num_batches, self.units_dim1, self.units_dim2)
        if self.use_bias:
            outputs = tf.add(outputs, self.bias)

        # Apply activation function
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):

        # Check if the input_shape is of rank 2 and that 
        # the last dimension is not None
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the input shape of a Dense layer '
                       'should be defined. Found None. '
                       f'Received: input_shape={input_shape}')

        return input_shape[:-1].concatenate([self.units_dim1, self.units_dim2])

    def get_config(self):

        # Call the method get_config of the superclass.
        config = super(Dense2Dto3D, self).get_config()

        # Update the config to add the additional parameters
        config.update({
            'units_dim1': self.units_dim1,
            'units_dim2': self.units_dim2,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        })

        return config

class Dense3Dto3D(tf.keras.layers.Layer):
    r"""An adaptation of the densely-connected NN layer that outputs a 3D tensor from a 3D tensor.

    :class:`Dense3Dto3D` is a change of the :class:`Dense2Dto3D` layer when the input is a tensor of order 3.
    It implements the dot product between the inputs and the kernel along the last axis of the inputs 
    and axis 1 of the kernel for each element in axis 1 of inputs and axis 0 of kernel : 
    
    >>> for d in range(nb_dense):
    >>>     output[:,d,:] = activation(tensordot(input[:,d,:], kernel[d,:,:]) + bias[d,:])
            
    It's like having several :class:`Dense` layers that have different inputs and which function independently.

    Parameters
    ----------
    units : int
        Dimensionality of the second dimension of the output space.

    activation : func or str, default:None
        Activation function to use. If you don't specify anything, 
        no activation is applied (ie. "linear" activation: `a(x) = x`).

    use_bias : boolean, default:True
        Indicates whether the layer uses a bias matrix.

    kernel_initializer : str or dict or func, default:'glorot_uniform'
        Initializer for the `kernel` weights tensor.

    bias_initializer : str or dict or func, default:'zeros'
        Initializer for the bias matrix.

    kernel_regularizer : str or dict or func, optional
        Regularizer function applied to the `kernel` weights tensor.

    bias_regularizer : str or dict or func, optional
        Regularizer function applied to the bias matrix.

    activity_regularizer : str or dict or func, optional
        Regularizer function applied to the output of the layer (its "activation").

    kernel_constraint : str or dict or func, optional
        Constraint function applied to the `kernel` weights tensor.

    bias_constraint : str or dict or func, optional
        Constraint function applied to the bias matrix.

    Input shape
    -----------
        3D tensor with shape: ``[batch_size, nb_dense, input_dim]``.

    Output shape
    ------------
        3D tensor with shape: ``[batch_size, nb_dense, units]``.
    """

    def __init__(self, 
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        # Superconstructor of the class tf.keras.layers.Layer
        super(Dense3Dto3D, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs)

        # Specify the dimension of the layer
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(f'Received an invalid value for `units_dim1`'
            f', expected positive integers. Received: units={units}')

        # Preprocess activation, bias, initialiers, regularizers, constraints
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Specify input shape
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        """
        Creates and initializes the kernel weight tensor and the bias 
        matrix of the layer. The rank of the input_shape needs to be 3.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the input.
        """

        # Check if the dtype is compatible
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('A Dense layer can only be built with'
                        f' a floating-point dtype. Received: dtype={dtype}')

        # Check if the input shape is compatible
        input_shape = tf.TensorShape(input_shape)
        nb_dense = tf.compat.dimension_value(input_shape[-2])
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None or nb_dense is None:
            raise ValueError('The two last dimension of the inputs to a Dense layer '
                        'should be defined. Found None. '
                        f'Full input shape received: {input_shape}')
        self.input_spec = InputSpec(ndim=3, axes={-2 : nb_dense, -1: last_dim})

        # Initialize kernel weights
        self.kernel = self.add_weight(
                        name='kernel',
                        shape=[nb_dense, last_dim, self.units],
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        dtype=self.dtype,
                        trainable=True)

        # Initialize bias weights
        if self.use_bias:
            self.bias = self.add_weight(
                            name='bias',
                            shape=[nb_dense, self.units],
                            initializer=self.bias_initializer,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                            dtype=self.dtype,
                            trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        """
        Defines the dot product between the kernel and the input independently
        fro each element of axis 1 of inputs and axis 0 of kernel.
        If `self.use_bias=True`, the output is summed with the bias matrix.
        If `self.activation` is not None, an activation function is applied 
        element-wise on the output. 

        Parameters
        ----------
        inputs : shape=(batch_size, nb_dense, input_dim)
            Input tensor.

        Returns
        -------
        outputs : shape=(batch_size, nb_dense, units)
            Output tensor.
        """

        # Change input dtype if incompatible with layer
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        # TODO : Compatibility with RaggedTensor and SparseTensor
        # TODO : Describe more precisely the operations done here

        # Tensor contraction between the inputs and the kernel on the axis i 
        # for each element of axis 1 of inputs and axis 0 of kernel: 
        # \forall d1, outputs_{b,d1,o} = \sum_{i} inputs_{b,d1,i} kernel_{d1,i,o} 
        # The abreviations are : 
        # b : batch
        # d : number of dense layers
        # i : last_dim of inputs
        # o : self.units
        # The shape of the input is (num_batches, nb_dense, last_dim) and the shape
        # output is (num_batches, nb_dense, self.units)
        outputs = tf.einsum('bdi,dio->bdo', inputs, self.kernel)

        # Add bias. The shape of outputs is (num_batches, nb_dense, self.units)
        # and of self.bias is (nb_dense, self.units). Consequently, broadcasting
        # is used whithin the sum to output a tensor of shape 
        # (num_batches, nb_dense, self.units)
        if self.use_bias:
            outputs = tf.add(outputs, self.bias)

        # Apply activation function
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):

        # Check if the input_shape is of rank 3 and that 
        # the last dimension is not None
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank(3)
        if tf.compat.dimension_value(input_shape[-1]) is None or tf.compat.dimension_value(input_shape[-2]) is None:
            raise ValueError('The two last dimension of the input shape of a Dense layer '
                       'should be defined. Found None. '
                       f'Received: input_shape={input_shape}')

        return input_shape[:-1].concatenate([self.units])

    def get_config(self):

        # Call the method get_config of the superclass.
        config = super(Dense3Dto3D, self).get_config()

        # Update the config to add the additional parameters
        config.update({
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        })

        return config

class Dense3Dto2D(tf.keras.layers.Layer):
    r"""An adaptation of the densely-connected NN layer that outputs a 2D tensor from a 3D tensor.

    :class:`Dense3Dto2D` is the inverse of the :class:`Dense2Dto3D` layer. It implements the dot product between 
    the inputs and the kernel along the two last axis of the inputs and the two first axis of the 
    kernel so that the inputs is projected in a 2D space 
    
    >>> output = activation(tensordot(input, kernel, axes=[[-2,-1], [0, 1]]) + bias). 
    
    It's like having :class:`Reshape` layer with a target shape of ``(input_dim1*input_dim2)`` followed by a :class:`Dense` 
    layer with ``units`` units.

    Parameters
    ----------
    units : int
        Dimensionality of the dimension of the output space.

    activation : func or str, default:None
        Activation function to use. If you don't specify anything, 
        no activation is applied (ie. "linear" activation: `a(x) = x`).

    use_bias : boolean, default:True
        Indicates whether the layer uses a bias vector.

    kernel_initializer : str or dict or func, default:'glorot_uniform'
        Initializer for the `kernel` weights tensor.

    bias_initializer : str or dict or func, default:'zeros'
        Initializer for the bias vector.

    kernel_regularizer : str or dict or func, optional
        Regularizer function applied to the `kernel` weights tensor.

    bias_regularizer : str or dict or func, optional
        Regularizer function applied to the bias vector.

    activity_regularizer : str or dict or func, optional
        Regularizer function applied to the output of the layer (its "activation").

    kernel_constraint : str or dict or func, optional
        Constraint function applied to the `kernel` weights tensor.

    bias_constraint : str or dict or func, optional
        Constraint function applied to the bias vector.

    Input shape
    -----------
        3D tensor with shape: ``(batch_size, input_dim1, input_dim2)``.

    Output shape
    ------------
        2D tensor with shape: ``(batch_size, units)``.
    """

    def __init__(self, 
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        # Superconstructor of the class tf.keras.layers.Layer
        super(Dense3Dto2D, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs)

        # Specify the dimension of the layer
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(f'Received an invalid value for `units`'
            f', expected positive integers. Received: units={units}')

        # Preprocess activation, bias, initialiers, regularizers, constraints
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Specify input shape
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        """
        Creates and initializes the kernel weight tensor and the bias 
        matrix of the layer. The rank of the input_shape needs to be 3.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the input.
        """

        # Check if the dtype is compatible
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('A Dense layer can only be built with'
                        f' a floating-point dtype. Received: dtype={dtype}')

        # Check if the input shape is compatible
        input_shape = tf.TensorShape(input_shape)
        input_dim1 = tf.compat.dimension_value(input_shape[-2])
        input_dim2 = tf.compat.dimension_value(input_shape[-1])
        if input_dim1 is None or input_dim2 is None:
            raise ValueError('The two last dimension of the inputs to a Dense layer '
                        'should be defined. Found None. '
                        f'Full input shape received: {input_shape}')
        self.input_spec = InputSpec(ndim=3, axes={-2 : input_dim1, -1: input_dim2})

        # Initialize kernel weights
        self.kernel = self.add_weight(
                        name='kernel',
                        shape=[input_dim1,  input_dim2, self.units],
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        dtype=self.dtype,
                        trainable=True)

        # Initialize bias weights
        if self.use_bias:
            self.bias = self.add_weight(
                            name='bias',
                            shape=[self.units],
                            initializer=self.bias_initializer,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                            dtype=self.dtype,
                            trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        """
        Defines the dot product between the kernel and the input on the two last
        axis of the inputs and the two first axis of the kernel.
        If `self.use_bias=True`, the output is summed with the bias matrix.
        If `self.activation` is not None, an activation function is applied 
        element-wise on the output. 

        Parameters
        ----------
        inputs : shape=(batch_size, input_dim1, input_dim2)
            Input tensor.

        Returns
        -------
        outputs : shape=(batch_size, units)
            Output tensor.
        """

        # Change input dtype if incompatible with layer
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        # Tensor contraction between the inputs and the kernel on the axis i and j : 
        # outputs_{b,o} = \sum_{i,j} inputs_{b,i,j} kernel_{i,j,o} 
        # The abreviations are : 
        # b : batch
        # i : input_dim1
        # j: input_dim2
        # o : self.units
        # The shape of the input is (num_batches, input_dim1, input_dim2) and the shape
        # output is (num_batches, self.units)
        outputs = tf.einsum('bij,ijo->bo', inputs, self.kernel)

        # Add bias. The shape of outputs is (num_batches, self.units)
        # and of self.bias is (self.units). Consequently, broadcasting
        # is used whithin the sum to output a tensor of shape 
        # (num_batches, self.units)
        if self.use_bias:
            outputs = tf.add(outputs, self.bias)

        # Apply activation function
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):

        # Check if the input_shape is of rank 3 and that 
        # the last dimension is not None
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank(3)
        if tf.compat.dimension_value(input_shape[-1]) is None or tf.compat.dimension_value(input_shape[-2]) is None:
            raise ValueError('The two last dimension of the input shape of a Dense layer '
                       'should be defined. Found None. '
                       f'Received: input_shape={input_shape}')

        return input_shape[:-2].concatenate([self.units])

    def get_config(self):

        # Call the method get_config of the superclass.
        config = super(Dense3Dto2D, self).get_config()

        # Update the config to add the additional parameters
        config.update({
            'units': self.units_dim1,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        })

        return config

class Dense3Dto4D(tf.keras.layers.Layer):
    r"""An adaptation of the densely-connected NN layer that outputs a 4D tensor from a 3D tensor.

    :class:`Dense3Dto4D` is the same adaptation from :class:`Dense` to :class:`Dense2Dto3D` layer but from the layer
    :class:`Dense3Dto3D` this time with a kernel of order 4. It implements the dot product between the 
    inputs and the kernel along the last axis of the inputs and axis 1 of the kernel for each 
    element in axis 1 of inputs and axis 0 of kernel : 

    >>> for d in range(nb_dense):
    >>>     output[:,d,:,:] = activation(tensordot(input[:,d,:,:], kernel[d,:,:,:]) + bias[d,:,:])
            
    It's like having several :class:`Dense` and :class:`Reshape` layers that have different inputs and which function 
    independently with ``units_dim1*units_dim2`` units followed by a :class:`Reshape` layer with a target shape 
    of ``(units_dim1, units_dim2)``.

    Parameters
    ----------
    units_dim1 : int
        Dimensionality of the first dimension of the output space.

    units_dim2 : int
        Dimensionality of the second dimension of the output space.

    activation : func or str, default:None
        Activation function to use. If you don't specify anything, 
        no activation is applied (ie. "linear" activation: `a(x) = x`).

    use_bias : boolean, default:True
        Indicates whether the layer uses a bias tensor.

    kernel_initializer : str or dict or func, default:'glorot_uniform'
        Initializer for the `kernel` weights tensor.

    bias_initializer : str or dict or func, default:'zeros'
        Initializer for the bias tensor.

    kernel_regularizer : str or dict or func, default:None
        Regularizer function applied to the `kernel` weights tensor.

    bias_regularizer : str or dict or func, default:None
        Regularizer function applied to the bias tensor.

    activity_regularizer : str or dict or func, default:None
        Regularizer function applied to the output of the layer (its "activation").

    kernel_constraint : str or dict or func, default:None
        Constraint function applied to the `kernel` weights tensor.

    bias_constraint : str or dict or func, default:None
        Constraint function applied to the bias tensor.

    Input shape
    -----------
        3D tensor with shape: ``(batch_size, nb_dense, input_dim)``.

    Output shape
    ------------
        4D tensor with shape: ``(batch_size, nb_dense, units_dim1, units_dim2)``.
    """

    def __init__(self, 
                 units_dim1,
                 units_dim2,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        # Superconstructor of the class tf.keras.layers.Layer
        super(Dense3Dto4D, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs)

        # Specify the dimension of the layer
        self.units_dim1 = int(units_dim1) if not isinstance(units_dim1, int) else units_dim1
        self.units_dim2 = int(units_dim2) if not isinstance(units_dim2, int) else units_dim2
        if self.units_dim1 < 0 or self.units_dim2 < 0:
            raise ValueError(f'Received an invalid value for `units_dim1` or `units_dim2`'
            f', expected positive integers. Received: units_dim1={units_dim1}, units_dim2=={units_dim2}')

        # Preprocess activation, bias, initialiers, regularizers, constraints
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Specify input shape
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        """
        Creates and initializes the kernel weight tensor and the bias 
        matrix of the layer. The rank of the input_shape needs to be 3.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the input.
        """

        # Check if the dtype is compatible
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('A Dense layer can only be built with'
                        f' a floating-point dtype. Received: dtype={dtype}')

        # Check if the input shape is compatible
        input_shape = tf.TensorShape(input_shape)
        nb_dense = tf.compat.dimension_value(input_shape[-2])
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None or nb_dense is None:
            raise ValueError('The two last dimension of the inputs to a Dense layer '
                        'should be defined. Found None. '
                        f'Full input shape received: {input_shape}')
        self.input_spec = InputSpec(ndim=3, axes={-2 : nb_dense, -1: last_dim})

        # Initialize kernel weights
        self.kernel = self.add_weight(
                        name='kernel',
                        shape=[nb_dense, last_dim, self.units_dim1, self.units_dim2],
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        dtype=self.dtype,
                        trainable=True)

        # Initialize bias weights
        if self.use_bias:
            self.bias = self.add_weight(
                            name='bias',
                            shape=[nb_dense, self.units_dim1, self.units_dim2],
                            initializer=self.bias_initializer,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                            dtype=self.dtype,
                            trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        """
        Defines the dot product between the kernel and the input independently
        fro each element of axis 1 of inputs and axis 0 of kernel.
        If `self.use_bias=True`, the output is summed with the bias matrix.
        If `self.activation` is not None, an activation function is applied 
        element-wise on the output. 

        Parameters
        ----------
        inputs : shape=(batch_size, nb_dense, input_dim)
            Input tensor.

        Returns
        -------
        outputs : shape=(batch_size, units_dim1, units_dim2)
            Output tensor.
        """

        # Change input dtype if incompatible with layer
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        # Tensor contraction between the inputs and the kernel on the axis i 
        # for each element of axis 1 of inputs and axis 0 of kernel: 
        # \forall d1, outputs_{b,d1,o,u} = \sum_{i} inputs_{b,d1,i} kernel_{d1,i,o,u} 
        # The abreviations are : 
        # b : batch
        # d : number of dense layers
        # i : last_dim of inputs
        # o : self.units_dim1
        # u : self.units_dim2
        # The shape of the input is (num_batches, nb_dense, last_dim) and the shape
        # output is (num_batches, nb_dense, self.units_dim1, self.units_dim2)
        outputs = tf.einsum('bdi,diou->bdou', inputs, self.kernel)

        # Add bias. The shape of outputs is (num_batches, nb_dense, self.units_dim1, self.units_dim2)
        # and of self.bias is (nb_dense, self.units_dim1, self.units_dim2). Consequently, broadcasting
        # is used whithin the sum to output a tensor of shape 
        # (num_batches, nb_dense, self.units_dim1, self.units_dim2)
        if self.use_bias:
            outputs = tf.add(outputs, self.bias)

        # Apply activation function
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):

        # Check if the input_shape is of rank 3 and that 
        # the last dimension is not None
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank(3)
        if tf.compat.dimension_value(input_shape[-1]) is None or tf.compat.dimension_value(input_shape[-2]) is None:
            raise ValueError('The two last dimension of the input shape of a Dense layer '
                       'should be defined. Found None. '
                       f'Received: input_shape={input_shape}')

        return input_shape[:-1].concatenate([self.units_dim1, self.units_dim2])

    def get_config(self):

        # Call the method get_config of the superclass.
        config = super(Dense3Dto4D, self).get_config()

        # Update the config to add the additional parameters
        config.update({
            'units_dim1': self.units_dim1,
            'units_dim2': self.units_dim2,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        })

        return config

# Add custom layers to tensorflow for serializing
get_custom_objects().update({'Dense2Dto3D': Dense2Dto3D})
get_custom_objects().update({'Dense3Dto3D': Dense3Dto3D})
get_custom_objects().update({'Dense3Dto2D': Dense3Dto2D})
get_custom_objects().update({'Dense3Dto4D': Dense3Dto4D})
