from __future__ import print_function

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from IPython import display
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

dataframe = pd.read_csv("broetchen.csv", sep=";")


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(learning_rate, steps, batch_size, input_features):
    """Trains a linear regression model.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      input_feature: Array of `string`s specifying columns from input data set
        to use as input feature.

    Returns:
        The trained linear regressor
    """

    periods = 10
    steps_per_period = steps / periods

    my_feature_data = dataframe[input_features].astype('float32')
    # my_label = "median_house_value"
    my_label = "bestellmenge"
    targets = dataframe[my_label].astype('float32')

    # Create input functions.
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column(input_feature) for input_feature in input_features]

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # Take a break and compute predictions.
        predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        # Compute loss.
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)

    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.plot(root_mean_squared_errors)
    plt.show()

    # Create a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)

    return linear_regressor


linear_regressor = train_model(
    learning_rate=0.05,
    steps=10,
    batch_size=1,
    input_features=["personen", "temperatur"]
)

print("Training finished, predict something...")


def input_fn_predict(persons, temperature):
    dataset = Dataset.from_tensor_slices({'personen': tf.constant([persons]), 'temperatur': tf.constant([temperature])})
    dataset = dataset.batch(1).repeat(1)
    return dataset.make_one_shot_iterator().get_next()


def predict(persons, temperature):
    predict_result = linear_regressor.predict(input_fn=lambda: input_fn_predict(persons, temperature))

    predict_result = np.array([item['predictions'][0] for item in predict_result])
    for i in predict_result:
        print("Predicted value for %0.2f persons at %0.2f degrees: %0.8f" % (persons, temperature, i))


predict(42, 7)
