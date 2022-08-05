import tensorflow as tf
from tensorflow import keras
import numpy as np
from purestochastic.model.layers import *
from purestochastic.model.activations import *
from purestochastic.model.base_uncertainty_models import *
from keras.layers import InputLayer, Dense, Input
from purestochastic.common.metrics import *

class DeepEnsembleModel(StochasticModel):
    """ Implementation of the DeepEnsemble model.

    The Deep Ensemble [1]_ is an ensemble of Deep Learning model trained independently and 
    combined for prediction in order to estimate uncertainty.

    The model can be constructed manually or it's possible to use the method ``toDeepEnsemble``
    to convert a simple :class:`keras.Model` object into a :class:`DeepEnsembleModel` object. 
    This class don't need specific loss function and can't use all of the tensorflow loss 
    function and also custom loss functions.

    Methods
    -------
    compute_loss(x=None, y=None, y_pred=None, sample_weight=None):
        Compute the loss independently for each model.
    _combine_predictions(predictions, stacked):
        Combine the predictions made by the models.
    compute_metrics(x, y, predictions, sample_weight):
        Specify the mean and stochastic part of the predictions to compute the metrics.
    predict(x):
        Compute the predictions of the model thanks to the `_combine_predictions` method.

    References
    ----------
    .. [1] Balaji Lakshminarayanan, Alexander Pritzel et Charles Blundell. « Simple and scalable
        predictive uncertainty estimation using deep ensembles ». In : Advances in Neural Information
        Processing Systems 2017-Decem.Nips (2017), p. 6403-6414. issn : 10495258. arXiv : 1612.01474.
    """

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """ Custom ``compute_loss`` function.

        This method overrides the ``compute_loss`` function so that the class doesn't 
        need specific loss function. It computes the loss for each model independently.

        Arguments
        ---------
        x : tf.Tensor
            Input data.
        y : tf.Tensor
            Target data.
        y_pred : tf.Tensor
            Predictions returned by the model (output of ``model(x)``)
        sample_weight : optional
           Sample weights for weighting the loss function.
        
        Returns
        -------
        The total loss.
        """

        # Define a function that computes the loss for one model
        def compute_loss_single_model(ytilde):

            return self.compiled_loss(y, ytilde, sample_weight, regularization_losses=self.losses)

        # Parallelization of the computation
        return tf.reduce_mean(tf.vectorized_map(compute_loss_single_model, tf.transpose(y_pred, (1,0) + tuple([i+2 for i in range(0, len(y_pred.shape)-2)]) )))

    def _combine_predictions(self, predictions, stacked):
        r""" Combine the predictions of all the models in order to quantify the uncertainty.
        
        This method combines the prediction of all the models in order to quantify uncertainty.
        The computation of uncertainty and the mean prediction is different according to the 
        structure of the network. For the moment, there are 2 possibilities (B = number of models): 

            * Mean Variance Activation (see method ``MeanVarianceActivation``)): 

                * Mean : :math:`\hat{\mu} = \dfrac{1}{B} \sum_{i=1}^{B} \hat{\mu}_i`
                * Epistemic Variance : :math:`\hat{\sigma}^2_{epi} = \dfrac{1}{B} \sum_{i=1}^{B} (\hat{y}_i - \hat{\mu})^2`
                * Aleatoric Variance : :math:`\hat{\sigma}^2_{alea} = \dfrac{1}{B} \sum_{i=1}^{B} (\sigma^2_i)`
            * No specific structure : 
            
                * Mean : :math:`\hat{y} = \dfrac{1}{B} \sum_{i=1}^{B} \hat{y}_i`
                * Variance : :math:`\hat{\sigma}^2 = \dfrac{1}{B} \sum_{i=1}^{B} (\hat{y}_i - \hat{y})^2`

        In the future, it will be possible to add other possibilities.

        Arguments
        ---------
        predictions : tf.Tensor
            Predictions returned by the model (output of ``model(x)``)
        stacked : boolean
            Boolean to indicate wheter the output should be stacked in a single tensor or not.

        Returns
        -------
        Predictions that have been combined. If ``stacked`` is True, the output is a one tensor.
        Otherwise, the output is a list of tensors.
        """

        # Case 1 : The Deep Ensemble outputs a variance and a mean for each model
        if self.layers[-1].get_config()['activation'] == 'MeanVarianceActivation':

            # Compute the mean accros the model
            mean_prediction = tf.reduce_mean(predictions[:,:,:,0], axis=1)

            # Compute the variance accros the model
            mean_variance_epistemic = tf.reduce_mean(tf.math.pow(predictions[:,:,:,0],2), axis=1) - tf.math.pow(mean_prediction,2)
            mean_variance_aleatoric = tf.reduce_mean(predictions[:,:,:,1], axis=1)
            mean_variance = mean_variance_epistemic + mean_variance_aleatoric

            if stacked == False:
                return mean_prediction, mean_variance
            else:
                return tf.stack((mean_prediction, mean_variance_epistemic, mean_variance_aleatoric), axis=-1).numpy()
        
        # Case 2 : The Deep Ensemble has a standard structure
        else:

            # Compute the mean accros the model
            mean_prediction = tf.reduce_mean(predictions, axis=1)

            # Compute the variance accros the model
            mean_variance = tf.reduce_mean(tf.math.pow(predictions,2), axis=1) - tf.math.pow(mean_prediction,2)

            if stacked == False:
                return mean_prediction, mean_variance
            else:
                return tf.stack((mean_prediction, mean_variance), axis=-1).numpy()

    def compute_metrics(self, x, y, predictions, sample_weight):
        """ Custom ``compute_metrics`` method.
        
        As stated in the parent method ``compute_metrics``, this method called the 
        parent function with the appropriate ``y_pred`` and ``stochastic_predictions`` 
        arguments.

        Arguments
        ---------
        x : tf.Tensor
            Input data.
        y : tf.Tensor
            Target data.
        predictions : tf.Tensor
            Predictions returned by the model (output of ``model(x)``)
        sample_weight : optional
           Sample weights for weighting the loss function.

        Returns
        -------
        See parent method.
        """

        y_pred, stochastic_predictions = self._combine_predictions(predictions, stacked=False)

        return super(DeepEnsembleModel, self).compute_metrics(x, y, y_pred, stochastic_predictions, sample_weight)

    def predict(self, x, **kwargs):
        """Combine predictions made by all the models.

        This method just called the parent's method and then combine predictions in order to quantify uncertainty.

        Arguments
        ----------
        x : tf.Tensor
            Input data.
        kwargs : optional
            Other Arguments of the `predict` parent's method.
        
        Returns
        -------
        np.ndarray
            Predictions made by the Deep Ensemble model.
        """

        # Compute the predictions
        predictions = super(DeepEnsembleModel, self).predict(x, **kwargs)

        return self._combine_predictions(predictions, stacked=True)

def toDeepEnsemble(net, nb_models):
    """Convert a regular model into a deep ensemble model.

    This method intends to be high-level interface to construct
    a Deep Ensemble model from a regular model. At present, only
    the densely-connected NN is compatible with a fully parallelizable 
    implementation. Other architecture are just concatenated models.

    Parameters
    ----------
    net : tf.keras.Sequential or tf.keras.Model
        a tensorflow model

    nb_models : int
        the number of models

    Return
    ------
    :class:`DeepEnsembleModel`
        a Deep Ensemble Model

    TODO
    ----
    Add support for other architectures
    """

    # If the model is not built, raises a ValueError to
    # ask to build the model
    if not net.built:
        raise ValueError(
          'This model has not yet been built. '
          'Build the model first by calling `build()` or by calling '
          'the model on a batch of data.')

    # Check whether the net is compatible with the fully parallelizable implementation
    is_compatible = np.all(list(map(lambda layer : isinstance(layer,(Dense, Dense2Dto3D,InputLayer)), net.layers)))

    if is_compatible:
        print(f'Your network is compatible with a fully parallelizable implementation.')

        # The net can come from a tf.keras.Sequential or tf.keras.Model.
        # The DeepEnsembleModel is a subclass of tf.keras.Model so it needs
        # an Input layer. That's why, if the original net is from tf.keras.Sequential
        # and doesn't contain an Input layer, we have to add it from the shape given
        # in the first layer.
        first_layer = net.layers[0]
        config = first_layer.get_config()
        if isinstance(first_layer, InputLayer):
            inputs = Input(**config)
            net.layers.pop(0)
        else:
            inputs = Input(shape=config["batch_input_shape"][1:], name = 'input')
        x = inputs

        # Iterate over all layers to convert them to the right type
        for layer in net.layers:
            config = layer.get_config()
            config['name'] = 'ensemble_' + config['name']
            # Convert Dense layer to Dense2Dto3D or Dense3Dto3D according to the input shape.
            if isinstance(layer, Dense):
                if len(x.shape)==2:
                    config['units_dim1'] = nb_models
                    config['units_dim2'] = config.pop('units')
                    x = Dense2Dto3D(**config)(x)
                elif len(x.shape)==3:
                    x = Dense3Dto3D(**config)(x)
            # Convert Dense2Dto3D to Dense3Dto4D.
            elif isinstance(layer, Dense2Dto3D):
                x = Dense3Dto4D(**config)(x)
        
        # Return and construct an instance of the DeepEnsembleModel
        return DeepEnsembleModel(inputs=inputs, outputs=x)

    else:
        print(f'Your network is not compatible with a fully parallelizable implementation. The'
        f'Deep Ensemble will just be a concatenation of the same model multiple times.')