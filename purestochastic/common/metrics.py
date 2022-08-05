import tensorflow as tf
from tensorflow import keras
from scipy.stats import norm
from keras.metrics import base_metric
from keras.utils.generic_utils import get_custom_objects

class PredictionIntervalCoverageProbability(base_metric.Mean):
    r"""Prediction Interval Coverage Probability metric.
    
    A prediction interval :math:`[\hat{\underline{y_i}}, \hat{\bar{y_i}}]` is constructed so that with 
    probability ``p``, :math:`y_i` is included in the interval. The PICP [1]_ (Prediction Interval Coverage 
    Probability) aims at computing the true percentage of values included in the interval. Mathematically,
    it is defined as : 

    .. math::
        PICP=\frac{1}{n} \displaystyle\sum_{i=1}^{n} c_j ~~\text{ avec } c_j = \left\{    \begin{array}{ll}
        1 &\text{ si }  y_i\in [\underline{\hat{y}_i} ,\overline{\hat{y}_i}]\\
        0 &\text{ si }  y_i \not \in [\underline{\hat{y}_i} ,\overline{\hat{y}_i}]
        \end{array}
        \right.
            
    The best PICP is a percentage that is equal to ``p``. If ``y`` is not a 1d array, the batch dimension needs 
    to be the first and the output value is the mean over all other dimensions. 

    Parameters
    ----------
    name : str, default: 'picp'
        String name of the metric instance.
    p : float, default: 0.95
        Probability of values included in the interval associated 
        with the predicted parameters of the gaussian distribution.
    input_type : {"gaussian", "pi"}, default: "gaussian"
        Type of input to the metric. If "gaussian", the metric assumes the predictions are given in
        the mean and the variance of a gaussian distribution. If "pi", the metric assumes the predictions are
        given in the lower and upper bound of the prediction interval.
        

    References
    ----------
    .. [1] Abbas Khosravi, Saeid Nahavandi et Doug Creighton. « A prediction interval-based ap-
        proach to determine optimal structures of neural network metamodels ». In : Expert Syst. Appl.
        37 (mars 2010), p. 2377-2387. doi : 10.1016/j.eswa.2009.07.059.
    """

    def __init__(self, name='picp', p=0.95, input_type="gaussian", **kwargs):
        super(PredictionIntervalCoverageProbability, self).__init__(name=name, **kwargs)

        self.p = p

        # Compute the quantile associated with p if input_type="gaussian" or set q_p to None if input_type="pi"
        self.q_p = norm.ppf(p + (1-p)/2) if input_type == "gaussian" else None

    def update_state(self, y_true, predictions, sample_weight=None):
        r"""Accumulates picp statistics.
        
        Parameters
        ----------
        y_true : shape= ``[batch size, d_0, ...,  d_N]``
            The ground truth values.
        predictions : shape= ``[batch size, d_0, ...,  d_N, 2]``
            The predicted values.
        sample_weight : optional
            Optional weighting of each example.
        Returns
        -------
            Update op.
        """

        # If input_type="gaussian", compute the prediction interval associated with the predicted parameters
        if self.q_p != None:
            
            # Separate the mean and the variance
            y_pred, var_pred = tf.unstack(predictions, axis=-1) 

            # Compute the bound of the prediction interval
            y_lower_pred = y_pred - self.q_p*tf.math.sqrt(var_pred)
            y_upper_pred = y_pred + self.q_p*tf.math.sqrt(var_pred)

        # Otherwise retrieve the prediction interval from the predictions
        else:

            # Separate the lower and upper bounds
            y_lower_pred, y_upper_pred = tf.unstack(predictions, axis=-1) 

        # Boolean array indicated wheter each value of y_true is in the interval
        in_out_interval = tf.cast(tf.math.logical_and(y_lower_pred <= y_true, y_true <= y_upper_pred), tf.float32)

        return super(PredictionIntervalCoverageProbability, self).update_state(in_out_interval, sample_weight=sample_weight)

class PredictionIntervalNormalizedAverageWidth(base_metric.Mean):
    r"""Prediction Interval Normalized Average Width metric.
    
    The PINAW [2]_ (Prediction Interval Normalized Average Width) computes the average 
    width of prediction intervals (:math:`[\hat{\underline{y_i}}, \hat{\bar{y_i}}]`) 
    normalized by a the distance between the maximum and minimum value of y. 
    Mathematically, it is defined as : 
    
    .. math::
        PINAW=\frac{1}{R*n}\displaystyle\sum_{i=1}^n[\overline{\hat{y}_i}- \underline{\hat{y}_i}] 

    avec :math:`R=\max(y)-\min(y)`.
    
    The best PINAW is the minimum value. If ``y`` is not a 1d array, the batch dimension needs 
    to be the first and the output value is the mean over all other dimensions.

    Parameters
    ----------
    name : str, default: 'pinaw'
        String name of the metric instance.
    p : float, default: 0.95
        Probability of values included in the interval associated 
        with the predicted parameters of the gaussian distribution.
    input_type : {"gaussian", "pi"}, default: "gaussian"
        Type of input to the metric. If "gaussian", the metric assumes the predictions are given in
        the mean and the variance of a gaussian distribution. If "pi", the metric assumes the predictions are
        given in the lower and upper bound of the prediction interval.

    References
    -----------
    .. [2] Abbas Khosravi et al. « Comprehensive Review of Neural Network-Based Prediction Intervals
        and New Advances ». In : IEEE Transactions on Neural Networks 22.9 (2011), p. 1341-1356.
        doi : 10.1109/TNN.2011.2162110.
    """

    def __init__(self, name='pinaw', p=0.95, input_type="gaussian", **kwargs):
        super(PredictionIntervalNormalizedAverageWidth, self).__init__(name=name, **kwargs)

        self.p = p

        # Compute the quantile associated with p if input_type="gaussian" or set q_p to None if input_type="pi"
        self.q_p = norm.ppf(p + (1-p)/2) if input_type == "gaussian" else None

    def update_state(self, y_true, predictions, sample_weight=None):
        r"""Accumulates pinaw statistics.
        
        Parameters
        ----------
        y_true : shape= ``[batch size, d_0, ...,  d_N]``
            The ground truth values.
        predictions : shape= ``[batch size, d_0, ...,  d_N, 2]``
            The predicted values.
        sample_weight : optional
            Optional weighting of each example.

        Returns
        -------
            Update op.
        """

        # If input_type="gaussian", compute the prediction interval associated with the predicted parameters
        if self.q_p != None:
            
            # Separate the mean and the variance
            y_pred, var_pred = tf.unstack(predictions, axis=-1) 

            # Compute the bound of the prediction interval
            y_lower_pred = y_pred - self.q_p*tf.math.sqrt(var_pred)
            y_upper_pred = y_pred + self.q_p*tf.math.sqrt(var_pred)

        # Otherwise retrieve the prediction interval from the predictions
        else:

            # Separate the lower and upper bounds
            y_lower_pred, y_upper_pred = tf.unstack(predictions, axis=-1) 

        # L1 Distance between the maximum and minimum value of y
        R = tf.math.reduce_max(y_true, axis=0) - tf.math.reduce_min(y_true, axis=0)

        # Compute the average width of all the prediction intervals
        normalized_width = (y_upper_pred - y_lower_pred)/R

        return super(PredictionIntervalNormalizedAverageWidth, self).update_state(normalized_width, sample_weight=sample_weight)

class CoverageWidthBasedCriterion(keras.metrics.Metric):
    r"""Coverage Width Based Criterion metric.

    The CWC [3]_ (Coverage Width Based Criterion) was defined to find a trade-off between 
    the PICP and the PINAW. Mathematically, it is defined as :

    .. math::
        CWC = PINAW(1+\gamma(PICP)e^{\eta(\mu-PICP)} ) ~~ \text{ avec } \gamma(PICP) = \left\{    \begin{array}{ll}
        1 &\text{ si } ~~ PICP < \mu\\
        0 &\text{ si }~~  PICP \geq \mu
        \end{array}
        \right.

    The operation can be explained in two parts : 

        * If the PICP is higher than p, the CWC is equaled to the PINAW metric so that the size of the prediction interval decreases.

        * If the PICP is lower than p, the exponential term becomes very large and the most influent criterion is the PICP that the size of the prediction interval increases.

    The best CWC is the minimum value. If ``y`` is not a 1d array, the batch dimension needs 
    to be the first and the output value is the mean over all other dimensions.

    Parameters
    ----------
    name : str, default: 'cwc'
        String name of the metric instance.
    p : float, default: 0.95
        Probability of values included in the interval associated 
        with the predicted parameters of the gaussian distribution.
    eta : positive float, default: 50
        Penalty value when the PICP is lower than p.
    input_type : {"gaussian", "pi"}, default: "gaussian"
        Type of input to the metric. If "gaussian", the metric assumes the predictions are given in
        the mean and the variance of a gaussian distribution. If "pi", the metric assumes the predictions are
        given in the lower and upper bound of the prediction interval.

    Warning
    --------
    If ``input_type="pi"``, it's important that the ``p`` specified in the constructor is the same as the one used to
    construct the prediction interval. Otherwise, the CWC will be wrong.

    References
    -----------
    .. [3] Abbas Khosravi et al. « Comprehensive Review of Neural Network-Based Prediction Intervals
        and New Advances ». In : IEEE Transactions on Neural Networks 22.9 (2011), p. 1341-1356.
        doi : 10.1109/TNN.2011.2162110.
    """

    def __init__(self, num_target, name='cwc', p=0.95, input_type="gaussian", eta=50, **kwargs):
        super(CoverageWidthBasedCriterion, self).__init__(name=name, **kwargs)

        # Store the parameters
        self.eta = eta
        self.p = p

        # Compute the quantile associated with p if input_type="gaussian" or set q_p to None if input_type="pi"
        self.q_p = norm.ppf(p + (1-p)/2) if input_type == "gaussian" else None

        # Add tf.Variable in order to store the metrics
        self.picp = self.add_weight(shape=(num_target,), name='picp', initializer='zeros')
        self.pinaw = self.add_weight(shape=(num_target,), name='pinaw', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, predictions, sample_weight=None):
        r"""Accumulates cwc statistics.
        
        Parameters
        ----------
        y_true : shape= ``[batch size, d_0, ... ,  d_N]``
            The ground truth values.
        predictions : shape= ``[batch size, d_0, ... ,  d_N, 2]``
            The predicted values.
        sample_weight : optional
            Optional weighting of each example.
        """

        # If input_type="gaussian", compute the prediction interval associated with the predicted parameters
        if self.q_p != None:
            
            # Separate the mean and the variance
            y_pred, var_pred = tf.unstack(predictions, axis=-1) 

            # Compute the bound of the prediction interval
            y_lower_pred = y_pred - self.q_p*tf.math.sqrt(var_pred)
            y_upper_pred = y_pred + self.q_p*tf.math.sqrt(var_pred)

        # Otherwise retrieve the prediction interval from the predictions
        else:

            # Separate the lower and upper bounds
            y_lower_pred, y_upper_pred = tf.unstack(predictions, axis=-1) 

        # Compute the PICP and the PINAW
        value_picp = tf.cast(tf.math.logical_and(y_lower_pred <= y_true, y_true <= y_upper_pred), tf.float32)
        self.picp.assign_add(tf.reduce_sum(value_picp, axis=0))
        self.pinaw.assign_add(tf.reduce_sum((y_upper_pred - y_lower_pred)/(tf.math.reduce_max(y_true, axis=0) - tf.math.reduce_min(y_true, axis=0)), axis=0))

        # Increase the counter with the batch_size
        self.count.assign_add(tf.cast(tf.size(value_picp), self._dtype))

    def result(self):

        return tf.reduce_mean((self.pinaw/self.count)*(1 + tf.cast(((self.picp/self.count) <= self.p), tf.float32)*tf.math.exp(-self.eta*((self.picp/self.count) - self.p))))

    def reset_state(self):

        self.picp.assign(tf.zeros(self.picp.get_shape()))
        self.pinaw.assign(tf.zeros(self.pinaw.get_shape()))
        self.count.assign(0)


get_custom_objects().update({'picp': PredictionIntervalCoverageProbability})
get_custom_objects().update({'pinaw': PredictionIntervalNormalizedAverageWidth})
get_custom_objects().update({'cwc': CoverageWidthBasedCriterion})