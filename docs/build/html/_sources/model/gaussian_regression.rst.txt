#######################
GaussianRegression
#######################

The gaussian regression model is an extension of the regression task. In regression, 
we try to predict the value of a target variable as a function of the values of
the independent variables. However, the uncertainty of the prediction is not taken
into account and it can be problematic during the modelisation. However, it is 
difficult to assess the quality of the uncertainty as this information is not really 
available. 

The uncertainty can be separated into two parts : 

    * ``Aleatoric uncertainty`` : the uncertainty linked to the distribution of the data
    * ``Epistemic uncertainty`` : the uncertainty related to the power of the model,
      the uncertainty related to the estimation of the model parameters and the uncertainty related to the collection of
      (incomplete, noisy, discordant, ...).

.. figure:: uncertainty.jpg
    :align: center
    :alt: uncetainty
    :width: 500px
    
    A schematic view of main differences between aleatoric and epistemic uncertainties from the article **A Review of Uncertainty Quantification in Deep Learning: Techniques, Applications and Challenges** [1]_

Modelling uncertainty can be done by modelling the distribution of the prediction.
It is not possible to predict the whole distribution of the prediction. One solution
is to predict some quantiles of the distribution and another solution is to predict
the parameters of a known distribution. In the case of :class:`GaussianRegression`,
this is the second solution and the model tries to predict the mean and the 
variance of a gaussian distribution associated with the prediction. 

.. warning::

    According to the model, it is possible to separate the variance into an 
    aleatoric and epistemic component. Therefore, the the number of outputs
    can be different.

.. tip::

    It is possible to use :class:`GaussianRegression` with a standard deterministic model.
    However, when you will call the method ``predict```, it will only send you the mean. It
    is useful to utilize the class to scale and unscale the data.

.. autoclass:: purestochastic.model.base_uncertainty_models.GaussianRegression
    :members:

.. raw:: html

   <h2>References</h2> 


.. [1] Moloud Abdar et al. « A review of uncertainty quantification in deep learning : Techniques,
    applications and challenges ». In : Information Fusion 76 (2021), p. 243-297. issn : 15662535.
    doi : 10.1016/j.inffus.2021.05.008. arXiv : 2011.06225.