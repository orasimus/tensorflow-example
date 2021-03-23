import numpy as np
import valohai as vh


vh.prepare(step='Preprocess data')

# Read input files from Valohai inputs directory
# This enables Valohai to version your training data
# and cache the data for quick experimentation

with np.load(vh.inputs('mnist').path(), allow_pickle=True) as file:
    x_train, y_train = file['x_train'], file['y_train']
    x_test, y_test = file['x_test'], file['y_test']

x_train, x_test = x_train / 255.0, x_test / 255.0


# Write output files to Valohai outputs directory
# This enables Valohai to version your data 
# and upload output it to the default data store

path = vh.outputs('dataset').path('preprocessed_mnist.npz')
np.savez(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
