import tensorflow as tf
from tensorflow import keras
import numpy as np
from purestochastic.model.layers import *
from purestochastic.model.base_uncertainty_models import *
from keras import backend as K
from purestochastic.common.regularizers import Orthonormality
from keras.layers import Dense

class OrthonormalCertificatesModel(StochasticModel):
    """ Implementation of the Orthonormal Certificates model.

    The model was proposed in [4]_ . To estimate epistemic uncertainty, they propose Orthonormal Certificates (OCs), 
    a collection of diverse non-constant functions that map all training samples to zero.

    The model can be constructed manually (not recommended) or it's possible to use the method ``toOrthonormalCertificates``
    to convert a simple :class:`keras.Model` object into a :class:`OrthonormalCertificatesModel` object. 
    
    Methods
    -------
    fit(X, y, epochs_oc=0, learning_rate_oc=0.001):
        Fit the initial and OC model.
    fit_oc(X, y, learning_rate_oc=0.001):
        Fit the OC model.
    compute_metrics(x, y, predictions, sample_weight):
        Specify the mean and stochastic part of the predictions to compute the metrics.
    predict(data, S=5, verbose=0):
        Computes the predictions of the initial model and an epistemic score.
    find_loss():
        Returns the loss specified in ``compile``.


    References
    ------------
    .. [4] Tagasovska, Natasa and Lopez-Paz, David. « Single-model uncertainties for deep learning ». 
        In : Advances in Neural Information Processing Systems 2019.Nips (2019), p. 1-12. issn : 10495258. 
        arXiv : 1811.00908.
    """

    def compute_metrics(self, x, y, predictions, sample_weight):
        """ Custom ``compute_metrics`` method.
        
        As stated in the parent method ``compute_metrics``, this method called the 
        parent function with the appropriate ``y_pred`` and ``stochastic_predictions`` 
        arguments. 

        Warning
        -------
        For ``OrthonormalCertificatesModel``, the choice is to remove stochastic
        metrics because the certificates don't have a real sense. 

        Arguments
        ---------
        x : tf.Tensor
            Input data.
        y : tf.Tensor
            Target data.
        predictions : tf.Tensor
            Predictions returned by the model (output of `model(x)`)
        sample_weight : optional
           Sample weights for weighting the loss function.

        Returns
        -------
        See parent method.
        """

        return super(StochasticModel, self).compute_metrics(x, y, predictions[0], sample_weight)

    def fit(self, X, y, epochs_oc=0, learning_rate_oc=0.001, **kwargs):
        """Train the model the initial model and the orthonormal certificates.

        The model is trained in two parts : 

        * During ``epochs`` epochs, the model is trained normally. It's defined as 
          the training of the initial model and the training uses the optimizer and 
          learning rate specified in the ``compile`` function. The certificates are
          frozen.

        * During ``epochs_oc`` epochs, all the layer are frozen except the certificates.
          The training is parametrized by ``learning_rate_oc`` and the sum of the loss 
          function specified in the ``compile`` function and the Orthonormality loss.

        Note
        -----
        By default, the parameter ``epochs_oc`` is set to 0, and the orthonormal certificates
        are not trained.

        See Also
        ---------
        purestochastic.common.regularizer.Orthonormality

        Parameters
        ----------
        X: np.ndarray
            The input data.
        y: np.ndarray
            The target data.
        epoch_oc : int (default : 0)
            Number of epochs for the training of certificates.
        learning_rate_oc : float (default : 0.001)
            Learning rate for the training of certificates.
        
        Returns
        -------
        History of the two trainings.
        """

        # Freeze OC layer
        self.layers[-1].trainable = False

        # Train the basic model
        self.compile(loss=[self.find_loss(), None], optimizer=self.optimizer, metrics=self.compiled_metrics._metrics, stochastic_metrics=self.stochastic_metrics)
        training_history = super(OrthonormalCertificatesModel, self).fit(X, y, **kwargs)

        # Train the OrthogonalCertificates model
        kwargs["epochs"] = epochs_oc
        training_history_oc = self.fit_oc(X, y, learning_rate_oc=learning_rate_oc, **kwargs)

        return training_history, training_history_oc

    def fit_oc(self, X, y, learning_rate_oc=0.001, **kwargs):
        """ Train the orthonormal certificates.

        All the layer are frozen except the orthonormal certificates. The model is trained 
        with the optimizer specified in the ``compile`` function with the learning rate
        ``learning_rate_oc``. The loss is the sum of the two following parts : 

            * The loss function with predicted value set to the output of the orthonormal
              certificates and target value set to 0.

            * The orthonormality regularizer added to the kernel so that the certificates 
              are orthonormal. For more details, see :class:``purestochastic.common.regularizer.Orthonormality``.

        The details of the method is detailled in [4]_.

        Parameters
        ----------
        X: np.ndarray
            The input data.
        y: np.ndarray
            The target data.
        learning_rate_oc : float (default : 0.001)
            Learning rate for the training of certificates.
        
        Returns
        -------
        History of the training.
        """

        #Freeze other layers and unfreeze OC layer
        for layer in self.layers[:-1]:
            layer.trainable = False
        self.layers[-1].trainable = True

        # Update the loss function to OC loss function and None for the first part
        K.set_value(self.optimizer.learning_rate, learning_rate_oc)
        self.compile(loss=[None, self.find_loss()], optimizer=self.optimizer, metrics=self.compiled_metrics._metrics, stochastic_metrics=self.stochastic_metrics)

        # Fit OC model
        y_oc = tf.zeros([X.shape[0], ] + self.output[1].shape[1:])
        training_history = super(OrthonormalCertificatesModel, self).fit(X, y_oc, **kwargs)

        #Unfreeze other layers and freeze OC layer
        for layer in self.layers[:-1]:
            layer.trainable = True
        self.layers[-1].trainable = False

        # Recompile the model to put the loss a the good place and save frozen layers
        self.compile(loss=[self.find_loss(), None], optimizer=self.optimizer, metrics=self.compiled_metrics._metrics, stochastic_metrics=self.stochastic_metrics)

        return training_history
        
    def predict(self, x, **kwargs):
        """ Compute predictions.

        This method just called the parent's method to compute the predictions of the initial model and
        the orthonormal certificates. The norm of the orthonormal certificates is computed in order to
        have a score for the epistemic uncertainty as defined in the article [4]_ .

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
        predictions, oc = super(OrthonormalCertificatesModel, self).predict(x, **kwargs)

        # Compute scores and normalize them
        scores = np.mean(np.power(oc, 2), axis=-1)
        # scores_epi = (scores - scores.min(axis=0)) / (scores.max(axis=0) - scores.min(axis=0))

        return tf.stack((predictions, scores), axis=-1).numpy()

    def find_loss(self):
        """ Returns the loss specified in the ``compile function``.

        Return
        ------
        str
            The name of the loss.
        """

        if isinstance(self.loss, str):
            return self.loss
        else:
            return [i for i in list(self.loss) if i != None][0]


def toOrthonormalCertificates(net, K, nb_layers_head, multiple_miso=True, lambda_coeff=1):
    """Convert a regular model into a Orthonormal Certificates model.

    This method intends to be high-level interface to construct
    a Orthonormal Certificates model from a regular model. 

    Parameters
    ----------
    net : :class:`tf.keras.Sequential` or :class:`tf.keras.Model`
        a tensorflow model

    nb_models : int
        the number of models

    Return
    ------
    :class:`class OrthonormalCertificatesModel`
        a Orthonormal Certificates Model
    """

    # Create Orthonormal Certificates for Epistemic Uncertainties
    input_oc = net.layers[-(nb_layers_head+1)].output
    if multiple_miso:
        nb_outputs = net.output.shape[1]
        output_oc = Dense2Dto3D(nb_outputs, K, name="output_oc", kernel_regularizer=Orthonormality(miso=False, lambda_coeff=lambda_coeff))(input_oc)
    else:
        output_oc = Dense(K, name="output_oc", kernel_regularizer=Orthonormality(lambda_coeff=lambda_coeff))(input_oc) 

    # Change output name of model
    net.layers[-1]._name = "output_initial_model"
        
    # Create the new model
    return OrthonormalCertificatesModel(net.input, [net.output, output_oc])
