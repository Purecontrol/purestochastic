import keras
import tensorflow as tf
import numpy as np
from purestochastic.model.base_uncertainty_models import StochasticModel
from purestochastic.model.deep_ensemble import toDeepEnsemble


class SWAGCallback(keras.callbacks.Callback):
    r"""Approximation of the posterior distribution of parameters as a gaussian distribution.

    Callback used in the class :class:`SWAGModel` and :class:`MultiSWAGModel`. It allows to approximate the 
    posterior distribution of the parameters as a gaussian distribution. The parameters of the
    gaussian distribution are computed as follows : 

    * The mean of the gaussian is the mean of the parameters (first moment) found during the training process. Mathematically, it is defined as : 

    .. math::

        \theta_{SWA} = \frac{1}{T} \sum_{t=1}^T \theta_t

    * The covariance matrix is constructed by taking half of a diagonal approximation and half of a low-rank approximation of the covariance matrix. The diagonal approximation is computed at the end of the training by using the first and second order moments of the parameters : 

    .. math::

        \Sigma_{Diag} = diag(\bar{\theta}^2-\theta_{SWA}^2)

    The low-rank approximation is constructed by using the difference of the last K values of the 
    parameters with the mean value of the parameters : 

    .. math::

        \Sigma_{low-rank} = \frac{1}{K-1}.\hat{D}\hat{D}^T \text{ avec chaque colonne de D } D_t=(\theta_t - \bar{\theta}_t)

    To sample from this gaussian distribution, the SWAGModel and MultiSWAGModel use the following equation : 

    .. math::

        \theta_j = \theta_{SWA} +\frac{1}{\sqrt{2}}.\Sigma_{diag}^{\frac{1}{2}}n_1 + \frac{1}{\sqrt{2(K-1)}}\hat{D}n_2, ~~ n_1, n_2 \sim \mathcal{N}(0,I)



    It is then sufficient to store the matrix D, the first order moments of the parameters as well as the 
    diagonal approximation of the covariance at the end of the training.

    Parameters
    -----------
    learning_rate : float
        The learning rate of the optimizer.
    update_frequency: int
        The number of epochs between two updates of the first and second moments of the parameters.
    K: int
        The number of samples used to compute the second order moments.
    """

    def __init__(self, learning_rate, update_frequency, K):
        super(SWAGCallback, self).__init__()

        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.K = K

    def on_epoch_end(self, epoch, logs=None):
        r"""Updates first and second order moments as well as deviation matrix.

        Every ``update_frequency`` epochs, the first and second order moments as well as deviation matrix are updated : 

        .. math::

            \bar{\theta} = \frac{n \bar{\theta} + \theta_{epochs}}{n+1}

        .. math::

            \bar{\theta}^2 = \frac{n \bar{\theta}^2 + \theta_{epochs}^2}{n+1}


        .. math::

            \text{APPEND_COL}(\hat{D}, \theta_{epochs}-\bar{\theta})
            


        If the matrix D has more than K columns, the oldest columns is removed.


        Parameters
        ----------
        epoch : int
            The number of the actual epoch.
        """

        # Initialize moments
        if epoch == 0:
            
            weights = self.model.get_weights()


            # Initialize order 2 moments, order 1 moments and deviation matrix
            self.order1_moments = []
            self.order2_moments = []
            self.deviation_matrix = []
            for i, array in enumerate(weights):
                array = array.reshape(-1)
                self.order1_moments.append(array)
                self.order2_moments.append(np.power(array,2))
                self.deviation_matrix.append((array - self.order1_moments[i]).reshape(-1,1))

        # Update moments by averaging the parameters
        elif epoch > 0:
            if epoch % self.update_frequency == 0:
                n = epoch/self.update_frequency
                
                weights = self.model.get_weights()
                for i, array in enumerate(weights):
                    array = array.reshape(-1)

                    # Update the moments
                    self.order1_moments[i] = (n*self.order1_moments[i] + array)/(n+1)
                    self.order2_moments[i] = (n*self.order2_moments[i] + np.power(array,2))/(n+1)

                    # Update the deviation matrix
                    if self.deviation_matrix[i].shape[1] >= self.K:
                        self.deviation_matrix[i] = np.delete(self.deviation_matrix[i], 0, axis=1)
                    self.deviation_matrix[i] = np.hstack((self.deviation_matrix[i], (array - self.order1_moments[i]).reshape(-1,1)))   
        
    def on_train_end(self, logs=None):
        """ Compute and store the variables needed to sample the posterior distribution.

        The mean of the gaussian distribution is saved in the attribute ``SWA_weights`` of 
        the model. The deviation matrix used in the covariance matrix is saved in the 
        attribute ``deviation_matrix`` of the model. Finally, the root of the diagonal 
        matrix used in the covariance matrix is computed and saved in the attribute
        ``SWA_cov`` of the model.

        Parameters
        ----------
        logs :  optional
            See tf.keras.callbacks.Callback
        """

        # Save the SWA weights
        self.model.SWA_weights = []
        for i, array in enumerate(self.order1_moments):
            self.model.SWA_weights.append(array)

        # Compute the element needed for the covariance matrix
        self.model.SWA_cov = []
        self.model.deviation_matrix = self.deviation_matrix
        for i, array in enumerate(self.order2_moments):
            self.model.SWA_cov.append( np.sqrt(np.maximum(array - np.power(self.model.SWA_weights[i],2),0))) 




class SWAGModel(StochasticModel):
    """ Implementation of the SWAG Model.

    The SWAG [2]_ (Stochastic Weight Averaging Gaussian) is a model to make bayesian inference and
    training to quantify uncertainty. For more details, see :class:`SWAGCallback`.

    The model can be constructed manually or it's possible to use the method `toSWAG`
    to convert a simple :class:`keras.Model` object into a :class:`SWAGModel` object. 

    Methods
    -------
    fit(X, y, start_averaging=10, learning_rate=0.001, update_frequency=1, K=10):
        Trains the model with the SWAG algorithm.
    _sample_prediction(data, S, verbose=0):
        Sample different prediction according to the posterior distribution of the parameters.
    _combine_predictions(predictions, stacked):
        Combine the sampled predictions.
    compute_metrics(x, y, predictions, sample_weight):
        Specify the mean and stochastic part of the predictions to compute the metrics.
    predict(data, S=5, verbose=0):
        Computes the predictions of the model with the SWAG algorithm.
    evaluate(x=None, y=None, S=5, sample_weight=None):
        Evaluate the model with the SWAG algorithm.


    References
    ----------
    .. [2] Wesley J. Maddox et al. « A simple baseline for Bayesian uncertainty in deep learning ». In :
        Advances in Neural Information Processing Systems 32.NeurIPS (2019), p. 1-25. issn : 10495258.
        arXiv : 1902.02476.
    """

    def fit(self, X, y, start_averaging=10, learning_rate=0.001, update_frequency=1, K=10, **kwargs):
        """Train the model with the SWAG algorithm.

        The model is trained in two parts : 

        * Before ``start_averaging`` epochs, the model is trained normally. It's defined as 
          the pretraining of the model and the training uses the optimizer and learning rate 
          specified in the ``compile`` function.

        * After ``start_averaging`` epochs, the model is trained with the SWAG callback. In other
          words, at the end of specific epochs (according to parameters), the parameters of the
          model are saved. At the end of the training, the callback computes the parameters of
          the approximated posterior gaussian distribution. The parameters are then used in 
          ``_sample_prediction`` in order to sample different predictions. At present, the optimizer 
          is necessarily the SGD optimizer. 

        See Also
        ---------
        src.model.swag.SWAGCallback

        Parameters
        ----------
        X: np.ndarray
            The input data.
        y: np.ndarray
            The target data.
        start_averaging: int
            The number of epochs to pretrain the model.
        learning_rate: float
            The learning rate of the SWAG algorithm (second part).
        update_frequency: int
            The number of epochs between each save of parameters of the SWAG algorithm.
        K: int
            The number of samples used to compute the covariance matrix.
        
        Returns
        -------
        History of the SWAG's training.
        """

        # Store the number of epochs given in kwargs and remove it from kwargs
        epochs = kwargs['epochs']
        del kwargs['epochs']

        # Make the pretraining of the model with the specified optimizer
        if kwargs.get("verbose") == 1:
            print("############ Pretraining ############")
        results_predict =  super(SWAGModel, self).fit(X, y, epochs=start_averaging, **kwargs)

        # Make the epochs with the SWAG strategy
        if kwargs.get("verbose") == 1:
            print("############ SWAG algorithm ############")
        self.compile(loss=self.loss, optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), metrics=self.compiled_metrics._metrics, stochastic_metrics=self.stochastic_metrics)
        results_predict =  super(SWAGModel, self).fit(X, y, epochs=epochs-start_averaging, callbacks=[SWAGCallback(learning_rate, update_frequency, K)], **kwargs)

        return results_predict

    def _sample_prediction(self, data, S, verbose=0):
        """Sample predictions according to the posterior distribution of the parameters.

        In the SWAG algorithm, the posterior distribution of the parameters is approximated
        as a Gaussian Distribution. The mean and the covariance are specified in the report
        associated with the code or in the article [2]_. The mean has been stored in the variable
        ``SWA_weights``. The diagonal and the Kth-rank approximation of the covariance matrix have
        been stored respectively in ``SWA_cov`` and ``deviation_matrix``.

        The method samples the weights and computes the prediction associated multiple times.

        Parameters
        ----------
        data : tf.Tensor
            Input data (equivalent to x). 
        S: int
            The number of samples used in the Monte Carlo method.
        verbose: int, default:0
            The verbosity level.

        Return
        ------
        preds : tf.Tensor
            The batch of S predictions.
        """

        final_weights = self.get_weights()

        # Set up random generator
        rng = np.random.default_rng()

        # Bayesian Model Averaging
        for s in range(S):

            weights_iter = []
            # Draw the value of the parameters
            for i, array in enumerate(final_weights):
                K = self.deviation_matrix[i].shape[1]
                weights_iter.append((self.SWA_weights[i] + self.SWA_cov[i]*rng.standard_normal(self.SWA_weights[i].shape[0])/np.sqrt(2) +  np.dot(self.deviation_matrix[i], rng.standard_normal(K))/np.sqrt(2*(K-1))).reshape(array.shape))
            
            # Set the weights for the iterations
            self.set_weights(weights_iter)

            # Compute the predictions
            pred_iter = super(SWAGModel, self).predict(data, verbose=verbose)

            # Initialize the predictions
            if s==0:
                preds = np.zeros((S,) + pred_iter.shape, dtype=np.float32)
            
            # Store the predictions
            preds[s] = pred_iter

        return preds

    def _combine_predictions(self, predictions, stacked):
        """ Bayesian Model Averaging of the S predictions.

        This method follows the ``_sample_prediction`` method. It takes in input the batch of S predictions
        sampled from ``_sample_prediction`` method. Then, it averages the predictions in order to compute
        the mean and the uncertainty associated with the prediction. The computation of uncertainty and the 
        mean prediction is different according to the structure of the network. For the moment, there are 2 
        possibilities (S=number of samples): 

        * Mean Variance Activation (see method ``MeanVarianceActivation``)): 

                * Mean : :math:`\hat{\mu} = \dfrac{1}{S} \sum_{i=1}^{S} \hat{\mu}_i`
                * Epistemic Variance : :math:`\hat{\sigma}^2_{epi} = \dfrac{1}{S} \sum_{i=1}^{S} (\hat{y}_i - \hat{\mu})^2`
                * Aleatoric Variance : :math:`\hat{\sigma}^2_{alea} = \dfrac{1}{S} \sum_{i=1}^{S} (\sigma^2_i)`
        * No specific structure

                * Mean : :math:`\hat{y} = \dfrac{1}{S} \sum_{i=1}^{S} \hat{y}_i`
                * Variance : :math:`\hat{\sigma}^2 = \dfrac{1}{S} \sum_{i=1}^{S} (\hat{y}_i - \hat{y})^2`

        In the future, it would be possible to add other possibilities.

        Parameters
        ----------
        predictions : tf.Tensor
            Batch of the S predictions computed by ``_sample_prediction``.
        stacked : boolean
            Boolean to indicate wheter the output should be stacked in a single tensor or not.
        """

        # Case 1 : The Deep Ensemble outputs a variance and a mean for each model
        if self.layers[-1].get_config()['activation'] == 'MeanVarianceActivation':

            mean_prediction = tf.reduce_mean(predictions[:,:,:,0], axis=0)

            mean_variance_epistemic = tf.reduce_mean(np.power(predictions[:,:,:,0],2), axis=0)  - tf.math.pow(mean_prediction,2)
            mean_variance_aleatoric = tf.reduce_mean(predictions[:,:,:,1], axis=0)
            mean_variance = mean_variance_epistemic + mean_variance_aleatoric

            if stacked == False:
                return mean_prediction, mean_variance
            else:
                return tf.stack((mean_prediction, mean_variance_epistemic,mean_variance_aleatoric), axis=-1).numpy()

        # Case 2 : The Deep Ensemble has a standard structure
        else:
            mean_prediction = tf.reduce_mean(predictions, axis=0)
            mean_variance = tf.math.reduce_variance(predictions, axis=0)

            if stacked == False:
                return mean_prediction, mean_variance
            else:
                return tf.stack((mean_prediction, mean_variance), axis=-1).numpy()

    def compute_metrics(self, x, y, predictions, sample_weight):
        """ Custom ``compute_metrics`` method.
        
        As stated in the parent method ``compute_metrics``, this method called the 
        parent function with the appropriate ``y_pred`` and ``stochastic_predictions`` 
        arguments.

        Warning
        -------
        Unless the model predicts aleatoric uncertainty, the model can't compute 
        stochastic metrics before the end of the training.

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

        if self.layers[-1].get_config()['activation'] == 'MeanVarianceActivation':

            y_pred = predictions[:,:,0]
            variance = predictions[:,:,1]

            return super(SWAGModel, self).compute_metrics(x, y, y_pred, variance, sample_weight)

        else:

            print("Warning : Impossible to compute stochastic metrics before the end of the training.")
            return super(StochasticModel, self).compute_metrics(x, y, predictions, sample_weight)

    def predict(self, data, S=5, verbose=0):
        """ Sample predictions and combine them.

        This method defines the inference step of the SWAG algorithm. First, it 
        samples predictions of the model with the ``_sample_prediction`` method. 
        Then, the predictions are combined with the method ``_combine_predictions``.

        Parameters
        ----------
        data: numpy.ndarray
            The input data.
        S: int, default:5
            The number of samples used in the Monte Carlo method.
        verbose: int, default:0
            The verbosity level.
        
        Returns
        -------
        The predictions of the model.
        """

        # Sample and compute the predictions
        predictions = self._sample_prediction(data, S, verbose=verbose)

        # Combine predictions
        return self._combine_predictions(predictions, stacked=True)

    def evaluate(self, x=None, y=None, S=5, sample_weight=None):
        """ Custom ``evaluate`` method.

        It returns the loss value & metrics values for the model in test mode.
        
        Parameters
        ----------
        x : tf.Tensor
            Input data.
        y : tf.Tensor
            Target data
        S : int, default:5
            The number of samples used in the Monte Carlo method.
        sample_weight : optional
           Sample weights for weighting the loss function.
        
        Return
        ------
        Dict containing the values of the metrics and loss of the model.
        """

        # Sample and compute the predictions
        predictions = self._sample_prediction(x, S, verbose=0)

        # Combine predictions
        y_pred, stochastic_predictions = self._combine_predictions(predictions, stacked=False)  

        return keras.utils.tf_utils.sync_to_numpy_or_python_type(super(SWAGModel, self).compute_metrics(x, tf.convert_to_tensor(y), y_pred, stochastic_predictions, sample_weight))


def toSWAG(net):
    """Convert a regular model into a SWAG model.

    This method intends to be high-level interface to construct
    a SWAG model from a regular model. At present, only
    the densely-connected NN is compatible with a fully parallelizable 
    implementation. Other architecture are just concatenated models.

    Parameters
    ----------
    net : :class:`tf.keras.Sequential` or :class:`tf.keras.Model`
        a tensorflow model

    nb_models : int
        the number of models

    Return
    ------
    :class:`SWAGModel`
        a SWAG Model
    """

    return SWAGModel.from_config(net.get_config())




class MultiSWAGModel(StochasticModel):
    """ Implementation of the MultiSWAG Model.

    The MultiSWAG [3]_ (Multi Stochastic Weight Averaging Gaussian) is an ensemble of 
    SWAG Model. It's a mix between a DeepEnsemble and SWAG Model. For more details, 
    see :class:`SWAGCallback`, :class:`SWAGModel` and :class:`DeepEnsembleModel`.

    The model can be constructed manually or it's possible to use the method ``toMultiSWAG``
    to convert a simple :class:`keras.Model` object into a `:class:MultiSWAGModel` object. This class don't
    need specific loss function and can't use all of the tensorflow loss function and also
    custom loss functions.

    Methods
    -------
    fit(X, y, start_averaging=10, learning_rate=0.001, update_frequency=1, K=10):
        Trains the model with the MultiSWAG algorithm.
    _sample_prediction(data, S, verbose=0):
        Sample different prediction according to the posterior distribution of the parameters.
    _combine_predictions(predictions, stacked):
        Combine the sampled predictions made by all models.
    compute_metrics(x, y, predictions, sample_weight):
        Specify the mean and stochastic part of the predictions to compute the metrics.
    evaluate(x=None, y=None, S=5, sample_weight=None):
        Evaluate the model with the MultiSWAG algorithm.
    predict(data, S=5, verbose=0):
        Computes the predictions of the model with the MultiSWAG algorithm.


    References
    ----------
    .. [3] Andrew Gordon Wilson et Pavel Izmailov. « Bayesian deep learning and a probabilistic
        perspective of generalization ». In : Advances in Neural Information Processing Systems 2020-
        Decem.3 (2020). issn : 10495258. arXiv : 2002.08791.
    """

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """ Custom ``compute_loss`` function.

        This method overrides the ``compute_loss`` function so that the class doesn't 
        need specific loss function. It computes the loss for each model independently.
        It's the same function as in :class:`DeepEnsembleModel`.

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

        def compute_loss_single_model(ytilde):

            return self.compiled_loss(y, ytilde, sample_weight, regularization_losses=self.losses)

        return tf.reduce_mean(tf.vectorized_map(compute_loss_single_model, tf.transpose(y_pred, (1,0) + tuple([i+2 for i in range(0, len(y_pred.shape)-2)]) )))

    def fit(self,X, y, start_averaging=10, learning_rate=0.001, update_frequency=1, K=10, **kwargs):
        """Train the model with the MultiSWAG algorithm.

        It's the same function as in :class:`SWAGModel` but with multiple models trained independently.
        The models are trained in two parts : 

        * Before ``start_averaging`` epochs, the models are trained normally. It's defined as 
          the pretraining of the models and the training uses the optimizer and learning rate 
          specified in the ``compile`` function.

        * After ``start_averaging`` epochs, the models are trained with the SWAG callback. In other
          words, at the end of specific epochs (according to parameters), the parameters of the
          models are saved. At the end of the training, the callback computes the parameters of
          the approximated posterior gaussian distribution. The parameters are then used in 
          ``_sample_prediction`` in order to sample different predictions. At present, the optimizer 
          is necessarily the SGD optimizer. For more details, see :class:`SWAGCallback`.

        Parameters
        ----------
        X: np.ndarray
            The input data.
        y: np.ndarray
            The target data.
        start_averaging: int
            The number of epochs to pretrain the model.
        learning_rate: float
            The learning rate of the MultiSWAG algorithm (second part).
        update_frequency: int
            The number of epochs between each save of parameters of the MultiSWAG algorithm.
        K: int
            The number of samples used to compute the covariance matrix.
        
        Returns
        -------
        History of the MultiSWAG's training.
        """

        # Store the number of epochs given in kwargs and remove it from kwargs
        epochs = kwargs['epochs']
        del kwargs['epochs']

        # Make the pretraining of the model with the specified optimizer
        if kwargs.get("verbose") == 1:
            print("############ Pretraining ############")
        results_predict =  super(MultiSWAGModel, self).fit(X, y, epochs=start_averaging, **kwargs)

        # Make the epochs with the SWAG strategy
        if kwargs.get("verbose") == 1:
            print("############ MultiSWAG algorithm ############")
        self.compile(loss=self.loss, optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), metrics=self.compiled_metrics._metrics, stochastic_metrics=self.stochastic_metrics)
        results_predict =  super(MultiSWAGModel, self).fit(X, y, epochs=epochs-start_averaging, callbacks=[SWAGCallback(learning_rate, update_frequency, K)], **kwargs)

        return results_predict

    def _sample_prediction(self,data, S, verbose=0):
        """Sample predictions according to the posterior distribution of the parameters.

        It's the same function as in :class:`SWAGModel`. In the MultiSWAG algorithm, the posterior 
        distribution of the parameters is approximated as a Gaussian Distribution. The mean 
        and the covariance are specified in the report associated with the code or in the 
        article. The mean has been stored in the variable ``SWA_weights``. The diagonal and the 
        Kth-rank approximation of the covariance matrix have been stored respectively in 
        ``SWA_cov`` and ``deviation_matrix``.

        The method samples the weights and computes the prediction associated multiple times
        for each model independently.

        Parameters
        ----------
        data : tf.Tensor
            Input data (equivalent to x). 
        S: int
            The number of samples used in the Monte Carlo method.
        verbose: int, default:0
            The verbosity level.

        Return
        ------
        preds : tf.Tensor
            The batch of S predictions.
        """

        final_weights = self.get_weights()

        # Set up random generator
        rng = np.random.default_rng()

        # Bayesian Model Averaging
        for s in range(S):

            weights_iter = []
            # Draw the value of the parameters
            for i, array in enumerate(final_weights):
                K = self.deviation_matrix[i].shape[1]
                weights_iter.append((self.SWA_weights[i] + self.SWA_cov[i]*rng.standard_normal(self.SWA_weights[i].shape[0])/np.sqrt(2) +  np.dot(self.deviation_matrix[i], rng.standard_normal(K))/np.sqrt(2*(K-1))).reshape(array.shape))
            
            # Set the weights for the iterations
            self.set_weights(weights_iter)

            # Compute the predictions
            pred_iter = super(MultiSWAGModel, self).predict(data, verbose=verbose)

            # Initialize the predictions
            if s==0:
                preds = np.zeros((S, ) + pred_iter.shape, dtype=np.float32)

            # Store the predictions
            preds[s] = pred_iter

        return preds

    def _combine_predictions(self, predictions, sampled, stacked):
        """ Bayesian Model Averaging of the S predictions of the B models.

        It's a little bit different from the function in :class:`SWAGModel`. There is 2 cases : 

        * If sampled is False, the parameters of the posterior distribution have not been computed
          yet and so it's impossible to sample predictions. Therefore, the function just combines
          the predictions made by all the models as in the :class:`DeepEnsembleModel`.

        * If sampled is True, the parameters have been computed. So, this method follows the 
          ``_sample_prediction`` method. It takes in input the batch of S predictions for each model 
          sampled from ``_sample_prediction`` method. Then, it averages the predictions over the samples and 
          the models in order to compute the mean and the uncertainty associated with the prediction. 
        
        The computation of uncertainty and the mean prediction is different according to the structure 
        of the network. For the moment, there are 2 possibilities (B = number of models, S = number of samples) : 

            * Mean Variance Activation (see method ``MeanVarianceActivation``)): 

                * Mean : :math:`\hat{\mu} = \dfrac{1}{B*S} \sum_{i=1}^{B} \sum_{j=1}^{S} \hat{\mu}_{i,j}`
                * Epistemic Variance : :math:`\hat{\sigma}^2_{epi} = \dfrac{1}{B*S} \sum_{i=1}^{B} \sum_{j=1}^{S} (\hat{y}_{i,j} - \hat{\mu})^2`
                * Aleatoric Variance : :math:`\hat{\sigma}^2_{alea} = \dfrac{1}{B*S} \sum_{i=1}^{B} \sum_{j=1}^{S} (\sigma^2_{i,j})`
            * No specific structure : 
            
                * Mean : :math:`\hat{y} = \dfrac{1}{B*S} \sum_{i=1}^{B} \sum_{j=1}^{S} \hat{y}_{i,j}`
                * Variance : :math:`\hat{\sigma}^2 = \dfrac{1}{B*S} \sum_{i=1}^{B} \sum_{j=1}^{S} (\hat{y}_{i,j} - \hat{y})^2`

        In the future, it would be possible to add other possibilities.

        Parameters
        ----------
        predictions : tf.Tensor
            Batch of the S predictions for each model computed by ``_sample_prediction``.
        sampled : boolean
            Boolean to indicate wheter the input have been sampled.
        stacked : boolean
            Boolean to indicate wheter the output should be stacked in a single tensor or not.
        """
        # Case 1 : The Deep Ensemble outputs a variance and a mean for each model
        if self.layers[-1].get_config()['activation'] == 'MeanVarianceActivation':

            average_axis = (0,2) if sampled == True else (1, )
            mean, variance = tf.unstack(predictions, axis=-1)

            mean_prediction = tf.reduce_mean(mean, axis=average_axis)
            mean_variance_epistemic = tf.reduce_mean(tf.math.pow(mean,2), axis=average_axis) - tf.math.pow(mean_prediction,2)
            mean_variance_aleatoric = tf.reduce_mean(variance, axis=average_axis)
            mean_variance = mean_variance_aleatoric + mean_variance_epistemic

            if stacked == False:
                return mean_prediction, mean_variance
            else:
                return tf.stack((mean_prediction, mean_variance_epistemic,mean_variance_aleatoric), axis=-1).numpy()
        # Case 2 : The Deep Ensemble has a standard structure
        else:
            average_axis = (0,2) if sampled == True else (1, )

            mean_prediction = tf.reduce_mean(predictions, axis=average_axis)
            mean_variance = tf.math.reduce_variance(predictions, axis=average_axis)

            if stacked == False:
                return mean_prediction, mean_variance
            else:
                return tf.stack((mean_prediction, mean_variance), axis=-1).numpy()

    def compute_metrics(self, x, y, prediction, sample_weight):
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
            Predictions returned by the model (output of `model(x)`)
        sample_weight : optional
           Sample weights for weighting the loss function.

        Returns
        -------
        See parent method.
        """

        y_pred, stochastic_predictions = self._combine_predictions(prediction, sampled=False, stacked=False)

        return super(MultiSWAGModel, self).compute_metrics(x, y, y_pred, stochastic_predictions, sample_weight)

    def predict(self, data, S=5, verbose=0):
        """ Sample predictions and combine them.

        It's the same function as in :class:`SWAGModel` This method defines the inference 
        step of the MultiSWAG algorithm. First, it samples predictions of each model 
        with the ``_sample_prediction`` method. Then, all the predictions are combined 
        with the method ``_combine_predictions``.

        Parameters
        ----------
        data: np.ndarray
            The input data.
        S: int, default:5
            The number of samples used in the Monte Carlo method.
        verbose: int, default:0
            The verbosity level.
        
        Returns
        -------
        The predictions of the model.
        """

        # Sample and compute the predictions
        predictions = self._sample_prediction(data, S, verbose=0)

        # Combine predictions
        return self._combine_predictions(predictions, sampled=True, stacked=True)

    def evaluate(self, x=None, y=None, S=5, sample_weight=None):
        """ Custom ``evaluate`` method.

        It returns the loss value & metrics values for the model in test mode.
        
        Parameters
        ----------
        x : tf.Tensor
            Input data.
        y : tf.Tensor
            Target data
        S : int, default:5
            The number of samples used in the Monte Carlo method.
        sample_weight : optional
           Sample weights for weighting the loss function.
        
        Return
        ------
        Dict containing the values of the metrics and loss of the model.
        """
        # Sample and compute the predictions
        predictions = self._sample_prediction(x, S, verbose=0)

        # Combine predictions
        y_pred, stochastic_predictions = self._combine_predictions(predictions, sampled=True, stacked=False)  

        return keras.utils.tf_utils.sync_to_numpy_or_python_type(super(MultiSWAGModel, self).compute_metrics(x, tf.convert_to_tensor(y), y_pred, stochastic_predictions, sample_weight))


def toMultiSWAG(net, nb_models):
    """Convert a regular model into a MultiSWAG model.

    This method intends to be high-level interface to construct
    a MultiSWAG model from a regular model. At present, only
    the densely-connected NN is compatible with a fully parallelizable 
    implementation. Other architecture are just concatenated models.

    Parameters
    ----------
    net : :class:`tf.keras.Sequential` or :class:`tf.keras.Model`
        a tensorflow model

    nb_models : int
        the number of models

    Return
    ------
    :class:`MultiSWAGModel`
        a MultiSWAG Model
    """

    deepEnsemble = toDeepEnsemble(net, nb_models)

    return MultiSWAGModel.from_config(deepEnsemble.get_config())
