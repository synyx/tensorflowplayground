from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import pandas as pd
from tensorflow import keras

# Helper libraries
import numpy as np

csv_data = pd.read_csv("broetchen.csv", sep=";")
csv_data["broetchen_verzehr"] = (csv_data["bestellmenge"] - csv_data["rest"])

model = keras.Sequential([
    keras.layers.Dense(1, input_dim=1)
])

model.compile(optimizer=keras.optimizers.SGD(lr=0.008, clipnorm=5.),
              loss='mean_squared_error',
              metrics=['accuracy'])

test_features = np.array(csv_data['personen'].astype('float32'))
test_labels = np.array(csv_data['broetchen_verzehr'].astype('float32'))

model.fit(test_features, test_labels, epochs=75,
          validation_split=0.08333333333)

test_loss, test_acc = model.evaluate(test_features, test_labels)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

result = model.predict(np.array([42]), batch_size=1)

print('Result:', result)