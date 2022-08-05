# Purestochastic

[![Documentation Status](https://readthedocs.org/projects/purestochastic/badge/?version=latest)](https://purestochastic.readthedocs.io/en/latest/?badge=latest)

Purestochastic lets you deal with the uncertainty associated with the prediction of your machine learning model. It makes you able to use all the tools of the [Tensorflow](https://www.tensorflow.org/) library and adds functionnality to be able to construct stochastic model easily. You can use a ``low-level`` interface by constructing model on your own way or a ``high-level`` interface by converting standard model into stochastic model.

> :warning: At present, the library is only constructed for the regression problem. At future, it will may be adapted to the classification problem.

## Table of Contents

- [Documentation](https://github.com/purecontrol/purestochastic#docs) : This section contains some ways to get few docs.
- [Examples](https://github.com/purecontrol/purestochastic#examples) :  This section will describe to you how you can use purestochastic.
- [Installation](https://github.com/purecontrol/purestochastic#installation) : This section will explain to you how can you install this library
- [Organisation](https://github.com/purecontrol/purestochastic#organisation) : This section will explain you the structure of the github repository.

## Documentation

The documentation is available at [here at readthedocs](https://purestochastic.readthedocs.io/en/latest/index.html). If an information is not available in the documentation, you can go directly in the source code.

## Examples

The library enables to convert standard deterministic ``keras.Model`` into ``stochastic model`` with high-level methods. Suppose you have constructed a standard model from keras API.

```python
inputs = Input(shape=(1,))
x = Dense(100, activation="relu")(inputs)
outputs = Dense(1)(x)
model = Model(inputs=inputs, outputs=outputs)
```

The library let you convert this model to a Deep Ensemble model with only one line of code.

```python
from purestochastic.model.deep_ensemble import toDeepEnsemble
deep_ensemble = toDeepEnsemble(model, nb_models=5)
deep_ensemble.summary()
    _________________________________________________________________
    Layer (type)                        Output Shape        Param #   
    =================================================================
    input (InputLayer)                  [(None, 1)]         0                                                            
    ensemble_hidden_layer (Dense2Dto3D)  (None, 5, 100)     1000                                                                                           
    ensemble_output (Dense3Dto3D)        (None, 5, 1)       505                                                                                                                        
    =================================================================
    Total params: 1,505
    Trainable params: 1,505
    Non-trainable params: 0
    _________________________________________________________________
```

The model has been converted and it is a now a Deep Ensemble. Thus, it is possible to quantify the uncertainty. For more examples, you can go the documentation in the [Examples section](https://purestochastic.readthedocs.io/en/latest/examples.html).

## Installation

Firstly you need to clone this repo on your computer with this git command : 

``` bash
git clone https://github.com/purecontrol/purestochastic.git
```

Then it's recommended to install dependencies using pip3 tool :

``` bash
pip3 install -r requirements.txt
```

Finally, you can install purestochastic with pip3 tool :

``` bash
pip3 install .
```

Then, you can use it in your code with this import :

```python
import purestochastic as ps
```

## Organisation

The repository is organized as follows : 

* `\docs` documentation files 
* `\examples` simple examples on how to use the library
* `\purestochastic` purestochastic package (each method or class is documented)
  * `\common` module with new losses and metrics
    * `losses.py` stochastic losses
    * `metrics.py` stochastic metrics
  * `model` tools to create stochastic models
    * `activations.py` custom activations
    * `layers.py` custom layers
    * `base_uncertainty_models.py` custom class StochasticModel and GaussianRegression
    * `deep_ensemble.py` tools for the Deep Ensemble model
    * `swag.py` tools for the SWAG and the MultiSWAG model
* `requirements.txt` all the package you need to install so that you can use the project on your own

