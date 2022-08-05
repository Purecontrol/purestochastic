.. purestochastic documentation master file, created by
   sphinx-quickstart on Wed Jul  6 15:46:26 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. .. automodule:: src.model.base_uncertainty_models
   :members:

Welcome to purestochastic's documentation!
==============================================

Purestochastic lets you deal with the uncertainty associated with the prediction. 
It makes you able to use all the tools of the `Tensorflow <https://www.tensorflow.org/>`_ 
library and adds functionnality to be able to construct stochastic model easily. 
You can use a ``low-level`` interface by constructing model on your own way or a 
``high-level`` interface by converting standard model into stochastic model.

.. toctree::
   :maxdepth: 1
   :caption: Guide

   installation.rst
   motivation.rst
   examples.rst

.. toctree::
   :maxdepth: 1
   :caption: Features

   model/model.rst
   utils/utils.rst

.. warning::
   At present, the library is only constructed for the regression problem. At future, 
   it can be adapted to the classification problem.
   
Purestochastic has its documentation hosted on Read the Docs.
