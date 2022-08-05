import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from purestochastic.model.swag import toSWAG
from purestochastic.model.base_uncertainty_models import GaussianRegression
from purestochastic.model.layers import Dense2Dto3D
from purestochastic.common.losses import gaussian_negative_log_likelihood
from purestochastic.model.activations import MeanVarianceActivation

#############################
######## TOY DATASET ########
#############################

# Generate the dataset
np.random.seed(12)
N = 20
X = np.random.rand(N,1)*8 -4 ; X_graph = np.linspace(-6, 6, 500).reshape(-1,1)
y = X**3 + np.random.randn(N,1)*3 ; y_graph = X_graph**3

# Plot the dataset
fig = plt.figure(figsize=(12,12))
plt.plot(X_graph, y_graph, 'b', label='y = x^3')
plt.scatter(X, y, s=60, c="r",label='Data points')
plt.legend(loc="lower right")
plt.show(block=False)

#############################
####### DEEP ENSEMBLE #######
#############################

# Specify input and output shape
input_shape = (1,) ; output_shape = 1

# Specify the initialisation of the weights
ki = RandomNormal(mean=0.0, stddev=0.3, seed=8888)

# Create one model
inputs = Input(shape=input_shape, name="input")
x = Dense(100, activation="relu", name="hidden_layer", kernel_initializer=ki)(inputs)
outputs = Dense2Dto3D(output_shape, 2, activation=MeanVarianceActivation, name="output", kernel_initializer=ki)(x)
model = Model(inputs=inputs, outputs=outputs)

# Convert to SWAG
swag = toSWAG(model)

print(swag.summary())

# Compile the model
swag.compile(loss=gaussian_negative_log_likelihood, optimizer=Adam(learning_rate=0.02))

# Create a GaussianRegression task
reg = GaussianRegression(swag)

# Train the model for 55 epochs
reg.fit(X, y, epochs=55, learning_rate=0.008, start_averaging=40)

# Make predictions
y_mean, y_var_epi, y_var_alea = reg.predict(X_graph, S=50)

# Plot the predictions
fig = plt.figure(figsize=(12,12))
plt.plot(X_graph, y_mean, 'g', label='Predicted')
plt.plot(X_graph, y_graph, 'b', label='Truth')
plt.fill_between(X_graph.reshape(-1), y_mean.reshape(-1) - 1.96*np.sqrt((y_var_alea).reshape(-1)), y_mean.reshape(-1) + 1.96*np.sqrt((y_var_alea).reshape(-1)), alpha=0.2, color='g', label="PI 95%")
plt.fill_between(X_graph.reshape(-1), y_mean.reshape(-1) - 1.96*np.sqrt((y_var_alea+y_var_epi).reshape(-1)), y_mean.reshape(-1) + 1.96*np.sqrt((y_var_alea+y_var_epi).reshape(-1)), alpha=0.2, color='g')
plt.scatter(X, y, s=60, c="r",label='Données')
plt.legend(loc="lower right", fontsize=20)
plt.tight_layout()
plt.show()