import tensorflow as tf
from tensorflow import keras
from keras.engine import compile_utils
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
import numpy as np
import pickle
import os
from purestochastic.model.activations import *


class StochasticModel(keras.Model, ABC):
    """ :class:`StochasticModel` allows to make stochastic training and inference features.

    :class:`StochasticModel` is a subclass of :class:`keras.Model` that allows to construct stochastic 
    model. Stochastic model often outputs the parameters of a parametric distribution or quantiles of a 
    generic distribution. For example, it outputs the mean and the variance of a Gaussian Distribution. 
    
    However, with standard class :class:`keras.Model`, all the metrics need to take the same input values. 
    Nevertheless, deterministic and stochastic metrics don't take the same input values. Therefore, 
    :class:`StochasticModel` adds the possibility to have deterministic as well as stochastic metrics. 
    Stochastic metrics need to be specified when ``model.compile`` is called with ``stochastic_metrics`` 
    or ``stochastic_weigthed_metrics`` arguments. 

    The class is abstract and can't be instanciate. Subclass need to override their own
    ``compute_metrics(self, x, y, prediction, sample_weight)`` method that will called
    the parent method with the appropriate `y_pred` and `stochastic_predictions` arguments.

    Methods
    -------
    compile(stochastic_metrics=None, stochastic_weigthed_metrics=None):
        Compile the model and add stochastic metrics.
    compute_metrics(x, y, y_pred, stochastic_predictions, sample_weight):
        Compute the values of the deterministic and stochastic metrics.
    reset_metrics():
        Reset the state of deterministic and stochastic metrics

    Warning
    --------
    All the stochastic metrics need to take the same input values. They have to be consistent
    together.     
    """

    def compile(self, stochastic_metrics=None, stochastic_weigthed_metrics=None, **kwargs):
        """ Configures the model for training.

        The method called the parent method ``compile`` and add additionnal variables
        for stochastic metrics compatibility.

        Parameters
        ----------
        stochastic_metrics : keras.metrics.Metric
            List of  stochastic metrics to be evaluated by the model during training
            and testing.
        stochastic_weigthed_metrics : keras.metrics.Metric
            List of  stochastic metrics to be evaluated and weighted by the model during 
            training and testing.
        **kwargs : 
            Arguments supported by ``compile`` parent method.
        """

        super(StochasticModel, self).compile(**kwargs)

        from_serialized = kwargs.pop('from_serialized', False)
        self.compiled_stochastic_metrics = compile_utils.MetricsContainer(stochastic_metrics, stochastic_weigthed_metrics, output_names=self.output_names, from_serialized=from_serialized)
        self.stochastic_metrics = self.compiled_stochastic_metrics._metrics if stochastic_metrics!=None else []
        self.stochastic_metrics_names = [metric.name for metric in self.stochastic_metrics] if stochastic_metrics!=None else []

    def compute_metrics(self, x, y, y_pred, stochastic_predictions, sample_weight):
        """ Compute the metrics.

        The method called the parent method ``compute_metrics`` to compute the
        deterministic metrics and then compute the stochastic metrics 
        manually. 
        
        The methods takes one additional parameter ``stochastic_predictions`` that it's
        specified by methods of subclass. This has to be the same for all the 
        stochastic metrics.
        
        Parameters
        ----------
        x : tf.Tensor
            Input data.
        y : tf.Tensor
            Target data.
        y_pred : tf.Tensor
            Mean prediction for y.
        stochastic_predictions : tf.Tensor
            Stochastic predictions for y.
        sample_weight : 
            Sample weights for weighting the metrics.

        Return
        ------
        metric_results : dict
            Value of each metric.
        """

        # Compute and collect deterministic metrics from y_pred
        metric_results = super(StochasticModel, self).compute_metrics(x, y, y_pred, sample_weight)

        # Compute stochastic metrics from y_pred, stochastic_predictions
        self.compiled_stochastic_metrics.update_state(y, tf.stack((y_pred, stochastic_predictions), axis=-1), sample_weight)

        # Collect results from stochastic metrics
        for metric in self.stochastic_metrics:
            if metric != None:
                result = metric.result()
                if isinstance(result, dict):
                    metric_results.update(result)
                else:
                    metric_results[metric.name] = result

        return metric_results

    def reset_metrics(self):
        """ Resets the state of all the metrics in the model.

        It's exactly the same as for :class:`keras.Model` except that it also resets the state 
        of stochastic metrics.
        """
        
        # Reset deterministic metrics
        for m in self.metrics:
            m.reset_state()

        # Reset stochastic metrics
        for m in self.stochastic_metrics:
            m.reset_state()

class Task(ABC):
    """
    This class is the parent class of all other classes that have been built for a specific task. 
    This class is abstract and therefore cannot be instantiated. Subclass need to implement the
    following methods : 
        * fit : model training
        * predict : model's prediction
        * evaluate : model's evaluation
        * save_weights : save weights of model
        * load_weights : load weights of model

    Models are for now : GaussianRegression, TimeSeriesModel

    Attributes
    ----------
    model : tf.keras.StochasticModel or tf.keras.Model or tf.keras.Sequential
        A statistic model.
    stochastic : boolean
        Specify if the model is stochastic
    """

    def __init__(self, model):
        
        # Store the model
        self.model = model

        # If the model is a subclass of StochasticModel, it outputs a variance.
        if issubclass(type(model), StochasticModel):
            self.stochastic = True
        else:
            self.stochastic = False

    @abstractmethod
    def fit(self, *args, **kwargs):
        """This method should implement the model training."""

    @abstractmethod
    def predict(self, *args, **kwargs):
        """This method sould compute the model's prediction"""

    @abstractmethod
    def evaluate(self, verbose, *args, **kwargs):
        """This method should implement evaluation of the model."""
    
    @abstractmethod
    def save_weights(self, *args, **kwargs):
        """This method should save all the layer weights"""

    @abstractmethod
    def load_weights(self, *args, **kwargs):
        """This method should load all the layer weights"""

class GaussianRegression(Task):
    """
    The class is designed for gaussian regression, in other words models that predict
    the mean and the variance associated of a gaussian distribution associated with
    the prediction. This class can work with deterministic and stochastic models.

    """

    def fit(self, X, y, standardize=True, **kwargs):
        """
        Train the model. It takes as input a matrix of input values X and a matrix of target values y.
        If standardize is set to true, the input values are standardized.

        Parameters
        ----------
        X : numpy.ndarray
            a matrix of input values
        y : numpy.ndarray
            a matrix of target values
        standardize : boolean
            specify if the input and the target values have to be scaled

        Returns
        -------
        A `History` object from the `fit` method of the model.
        """

        # If standardize is set to true, standardize input and target values
        self.standardize = standardize
        if self.standardize:
            self.scalerX = StandardScaler()
            X = self.scalerX.fit_transform(X)
            self.scalerY = StandardScaler()
            y = self.scalerY.fit_transform(y)

        # Fit the model
        return self.model.fit(X, y, **kwargs)
    
    def predict(self, X, **kwargs):
        """
        Computes the model's prediction. It takes as input a matrix of input values X.
        It outputs predictions made on the input values.

        Parameters
        ----------
        X : numpy.ndarray
            a matrix of input values  

        Returns
        -------
        It always outputs a matrix for the mean predictions made on the input values. 
        It can also output the variance if the model output the variance.
        It type_var=="sum", the variance is the sum of the aleatoric and epistemic variances. 
        Otherwise, the two types are returned separately.
        """

        # If standardize is set to true, standardize input values
        if self.standardize:
            X = self.scalerX.transform(X)

        ## Make predictions
        preds = self.model.predict(X, **kwargs)

        # Unstandardize target values (different case according to the activation function)
        return self.unstandardize_prediction(*np.moveaxis(preds, -1, 0))  # equivalent to tf.unstack(preds, axis=-1)
 
    def unstandardize_prediction(self, mean , variance_epi=None, variance_alea=None):
        """
        The goal of this method is to unstandardize the prediction wheter it's the variance
        or the mean of the prediction.

        Parameters
        ----------
        mean : np.array
            The predicted mean.
        variance_epi : np.array (optional)
            The predicted epistemic variance.
        variance_alea : np.array (optional)
            The predicted aleatoric variance.

        Return
        ------
        The prediction which have been unstandardize.
        """

        if self.standardize:

            output_values = (self.scalerY.inverse_transform(mean) ,)

            if not(variance_epi is None):
                output_values += (variance_epi*(self.scalerY.scale_**2),)

            if not(variance_alea is None):
                output_values += (variance_alea*(self.scalerY.scale_**2),)

        else:

            output_values = (mean, )

            if not(variance_epi is None):
                output_values += (variance_epi,)

            if not(variance_alea is None):
                output_values += (variance_alea,)

        return output_values

    def evaluate(self, X, y, metrics=None, stochastic_metrics=None, **kwargs):
        """
        Evaluate the model. It takes as input a matrix of input values X and a matrix of target values y.
        It computes different metrics on the model's predictions and the target values.
        If evaluation metrics are not specified, the function uses the metrics given in the `compile` method
        of the model. Metrics can be a list of metrics or a single metric and need to be separated between
        deterministic metrics and stochastic metrics. For more information, see StochasticModel.

        Parameters
        ----------
        X : numpy.ndarray
            a matrix of input values
        y : numpy.ndarray
            a matrix of target values
        metrics : keras.metrics.Metric
            a list of metrics to be computed on the model's predictions and the target values.
        stochastic_metrics : keras.metrics.Metric
            a list of metrics to be computed on the model's predictions and the target values.

        Returns
        -------
        A dictionary of metrics.
        """

        if metrics==None and stochastic_metrics==None:

            return self.model.evaluate(self.scalerX.transform(X),self.scalerY.transform(y))

        else:

            y = tf.convert_to_tensor(y, dtype=tf.float32)

            # Make predictions for input values
            preds = self.predict(X, **kwargs)
            mean_prediction = tf.convert_to_tensor(preds[0], dtype=tf.float32)
            if len(preds) == 3:
                variance_prediction = tf.convert_to_tensor(preds[1] + preds[2], dtype=tf.float32)
            elif len(preds) == 2:
                variance_prediction = tf.convert_to_tensor(preds[1], dtype=tf.float32)

            # # If evaluation_metrics is not a list, convert it into list
            # if not isinstance(evaluation_metrics,list):
            #     evaluation_metrics = [evaluation_metrics]
            
            # Compute predictions
            compiled_metrics = compile_utils.MetricsContainer(metrics)
            compiled_metrics.update_state(y, mean_prediction) 
            metrics = compiled_metrics._metrics[0] if metrics!=None else []

            if self.stochastic:
                compiled_stochastic_metrics = compile_utils.MetricsContainer(stochastic_metrics)
                compiled_stochastic_metrics.update_state(y, tf.stack([mean_prediction, variance_prediction], axis=-1))
                if stochastic_metrics!=None:
                    metrics += compiled_stochastic_metrics._metrics[0]
            
            return_metrics = {}
            for metric in metrics:
                result = metric.result().numpy()
                if isinstance(result, dict):
                    return_metrics.update(result)
                else:
                    return_metrics[metric.name] = result

            return return_metrics

    def save_weights(self, filepath, **kwargs):
        """
        Saves the model's weights.

        Parameters
        ----------
        filepath : string
            the path where the weights needs to be saved.
        **kwargs : dict
            a dictionary of parameters to be passed to the `save_weights` method of the tf.keras.Model.
        """
        # Save the model's weights
        self.model.save_weights(filepath, **kwargs)

        # Save mean and variance of the standardization parameters
        if self.standardize:
            with open(filepath+"scalerX.pkl", "wb") as f:
                pickle.dump(self.scalerX, f)
            with open(filepath+"scalerY.pkl", "wb") as f:
                pickle.dump(self.scalerY, f)

    def load_weights(self, filepath,**kwargs):
        """
        Loads the model's weights.

        Parameters
        ----------
        filepath : string
            the path where the weights are saved.
        **kwargs : dict
            a dictionary of parameters to be passed to the `load_weights` method of the tf.keras.Model.
        """

        # Load mean and variance of the standardization parameters

        #know if the file filepath+"scalerX.pkl" exists
        if os.path.isfile(filepath+"scalerX.pkl"):
            self.standardize = True
            with open(filepath+"scalerX.pkl", "rb") as f:
                self.scalerX = pickle.load(f)
            with open(filepath+"scalerY.pkl", "rb") as f:
                self.scalerY = pickle.load(f)

        # Load the model's weights
        self.model.load_weights(filepath, **kwargs)

