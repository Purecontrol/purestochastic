import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from purestochastic.model.orthonormal_certificates import toOrthonormalCertificates
from purestochastic.model.base_uncertainty_models import GaussianRegression

#############################
######## TOY DATASET ########
#############################

# Generate the dataset
np.random.seed(12)
N = 20
X = np.random.rand(N,1)*8 -4 ; X_graph = np.linspace(-6, 6, 500).reshape(-1,1)
y = X**3 + np.random.randn(N,1)*3 ; y_graph = X_graph**3

########################################
####### Orthonormal Certificates #######
########################################

# Specify input and output shape
input_shape = (1,) ; output_shape = 1

# Create one model
inputs = Input(shape=input_shape, name="input")
x = Dense(100, activation="relu", name="hidden_layer")(inputs)
outputs = Dense(output_shape, name="output")(x)
model = Model(inputs=inputs, outputs=outputs)

# Convert to Orthonormal Certificates
K=100
orthonormal_certificates = toOrthonormalCertificates(model, K=K, nb_layers_head=1)

print(orthonormal_certificates.summary())

# Compile the model
orthonormal_certificates.compile(loss="mse", optimizer=Adam(learning_rate=0.1))

# Create a GaussianRegression task
reg = GaussianRegression(orthonormal_certificates)

# Train the model for 40 epochs and train the orthonormal certificates for 40 epochs
reg.fit(X, y, epochs=40, epochs_oc=40)

# Make predictions
y_mean, score_epi = reg.predict(X_graph)

# Plot the predictions
fig = plt.figure(figsize=(12,12))
plt.plot(X_graph, y_mean, 'g', label='Predicted')
plt.plot(X_graph, y_graph, 'b', label='Truth')
plt.fill_between(X_graph.reshape(-1), y_mean.reshape(-1) - 60*np.sqrt(score_epi.reshape(-1)), y_mean.reshape(-1) + 60*np.sqrt(score_epi.reshape(-1)), alpha=0.2, color='g', label="Epistemic Score")
plt.scatter(X, y, s=60, c="r",label='Data point')
plt.legend(loc="lower right", fontsize=20)
plt.tight_layout()
plt.show()

