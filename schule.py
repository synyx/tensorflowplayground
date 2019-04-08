import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)

# 3. optimieren
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

# 1. daten laden, trainieren

csv_data = pd.read_csv("broetchen.csv", sep=";")
print(csv_data.describe())


def input_fn():
    features = {'personen': csv_data['personen'].astype('float32')}
    labels = csv_data['bestellmenge'].astype('float32').astype('float32')

    dataset = Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(12).repeat(1)

    return dataset.make_one_shot_iterator().get_next()


feature_columns = [tf.feature_column.numeric_column('personen')]


#linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns)
linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=optimizer)

linear_regressor.train(input_fn)

# 2. vorhersage

def input_fn_predict():
    dataset = Dataset.from_tensor_slices({'personen': tf.constant([42])})
    dataset = dataset.batch(1).repeat(1)
    return dataset.make_one_shot_iterator().get_next()


predict_result = linear_regressor.predict(input_fn_predict)
predict_result = np.array([item['predictions'][0] for item in predict_result])

print(predict_result[0])