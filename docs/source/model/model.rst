################
     Model
################

This part of the package contains the stochastic models as well as class for achieving specific goals.


.. raw:: html

   <h2>Tasks</h2>

The first part is the tasks. At present, there is only one task but it will be interesting to add more
of them in the future like quantile regression or binary classification. The aim of the task is to take
one stochastic model as **attribute** and and make all the **preprocessing** and **postprocessing** steps 
needeed for training and testing the models. For instance, :class:`GaussianRegression` allows the user to
scale the input data and unscale the predicted mean and variance.

Here is the list of tasks : 

.. toctree:: 
   :maxdepth: 1
   
   gaussian_regression.rst


.. raw:: html

   <h2>Modules</h2> 

The second part is the modules. The modules are the building blocks of the stochastic models.
Custom layers, activations and models are implemented in order to construct stochastic models.
The layers are used to construct the stochastic models and the activations can be used by users
to specify their own models dedicated for their specific needs. For more details, an example 
of utilization of each model is provided in their documentation.

Here is the list of modules : 

.. toctree:: 
   :maxdepth: 3
   
   custom_layers.rst
   custom_activations.rst
   custom_models.rst
