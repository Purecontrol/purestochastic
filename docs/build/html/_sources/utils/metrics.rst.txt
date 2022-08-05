################
     Metrics
################

Implementation of new stochastic metrics. The metrics assume that the predictions are the 
parameters of a gaussian distribution. These metrics have to be specified with the arguments
``stochastic_metrics`` or ``stochastic_weighted_metrics`` when calling the method ``compile``
in a subclass of :class:`StochasticModel` : 

>>> from src.common.metrics import PredictionIntervalCoverageProbability
>>> from src.model.base_uncertainty_models import StochasticModel
>>> issubclass(model, StochasticModel)
True
>>> model.compile(stochastic_metrics=[PredictionIntervalCoverageProbability()])

The metrics :class:`PICP`, :class:`PINAW` and :class:`CWC` suppose predictions given in ``update_state`` 
are different according the argument ``input_type`` : 
    
   1. If ``input_type="gaussian"``, the predictions need to be the means and the variances of a gaussian distribution defined as :

    * :math:`\hat{\mu} = \text{predictions}[ : , \cdots , :  , 0]`

    * :math:`\hat{\sigma}^2 = \text{predictions}[ : , \cdots , : , 1]`

   2. If ``input_type="pi"``, the predictions need to be the lower and upper bounds of the predictions defined as : 

    * :math:`\hat{y}_{lower} = \text{predictions}[ : , \cdots , :  , 0]`

    * :math:`\hat{y}_{upper} = \text{predictions}[ : , \cdots , : , 1]`

Here is the list of the new metrics :

.. contents:: :local:
    :depth: 2

PICP
-----

.. autoclass:: purestochastic.common.metrics.PredictionIntervalCoverageProbability
    :members:

PINAW
-------

.. autoclass:: purestochastic.common.metrics.PredictionIntervalNormalizedAverageWidth
    :members:

CWC
----

.. autoclass:: purestochastic.common.metrics.CoverageWidthBasedCriterion
    :members: update_state
