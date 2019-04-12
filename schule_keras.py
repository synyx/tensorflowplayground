from __future__ import absolute_import, division, print_function
import pandas as pd
from tensorflow import keras
import numpy as np

# 1. daten importieren
csv_data = pd.read_csv("broetchen.csv", sep=";")
csv_data["broetchen_verzehr"] = (csv_data["bestellmenge"] - csv_data["rest"])

# 2. modell erzeugen & kompilieren
model = keras.Sequential([
    keras.layers.Dense(1, input_dim=1)
])

model.compile(optimizer=keras.optimizers.SGD(lr=0.008, clipnorm=5.),
              loss='mean_squared_error',
              metrics=['accuracy'])

# 3. features & labels bereitstellen
test_features = np.array(csv_data['personen'].astype('float32'))
test_labels = np.array(csv_data['broetchen_verzehr'].astype('float32'))

# 4. modell trainieren
model.fit(test_features, test_labels, epochs=75,
          validation_split=0.08333333333)

# 5. dinge vorhersagen
result = model.predict(np.array([42]), batch_size=1)

print('Result:', result)

# 6. zweites feature
#test_features = [np.array(csv_data['personen'].astype('float32')), np.array(csv_data['temperatur'].astype('float32'))]
#result = model.predict((np.array([42]), np.array([7])), batch_size=1)