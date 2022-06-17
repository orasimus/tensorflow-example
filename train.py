import numpy as np
import tensorflow as tf
import valohai as vh


vh.prepare(step='Train model')

# Read input files from Valohai inputs directory
# This enables Valohai to version your training data
# and cache the data for quick experimentation

with np.load(vh.inputs('preprocessed_mnist').path(), allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=vh.parameters('learning_rate').value)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Print metrics out as JSON
# This enables Valohai to version your metadata
# and for you to use it to compare experiments

def log(epoch, logs):
    with vh.logger() as logger:
        logger.log('epoch', epoch)
        logger.log('accuracy', logs['accuracy'])
        logger.log('loss', logs['loss'])


cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=log)

model.fit(x_train, y_train, epochs=vh.parameters('epochs').value)


# Evaluate the model and print out the test metrics as JSON

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
with vh.logger() as logger:
    logger.log('test_accuracy', test_acc)
    logger.log('test_loss', test_loss)


# Write output files to Valohai outputs directory
# This enables Valohai to version your data 
# and upload output it to the default data store

model.save(vh.outputs('model').path('model.h5'))
