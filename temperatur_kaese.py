import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(101)
tf.set_random_seed(101)

# Read the training data
dataframe = pd.read_csv("temperature_kaese.csv", sep=";")
y_temperatur = dataframe["temperatur_durchschnitt"].values
x_verbrauch_kaese = dataframe["verbrauch_kaese_kg"].values

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(np.random.randn(), name="W")
b = tf.Variable(np.random.randn(), name="b")

number_of_data_points = len(x_verbrauch_kaese)

# hypothesis
y_pred = tf.add(tf.multiply(X, W), b)

# mean squared error
cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * number_of_data_points)

learning_rate = 0.01
training_epochs = 1000
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # init variables
    sess.run(init)

    for epoch in range(training_epochs):
        # feeding the optimizer
        for (_x, _y) in zip(x_verbrauch_kaese, y_temperatur):
            sess.run(optimizer, feed_dict={X: _x, Y: _y})

        training_cost = sess.run(cost, feed_dict={X: _x, Y: _y})
        weight = sess.run(W)
        bias = sess.run(b)
        print('Epoch',epoch)
        print(training_cost)
        print(weight)
        print(bias)
        print()

predictions = weight * x_verbrauch_kaese + bias
print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n')

plt.plot(x_verbrauch_kaese, y_temperatur, 'ro', label ='Original data')
plt.plot(x_verbrauch_kaese, predictions, label ='Fitted line')
plt.ylabel('Temperatur')
plt.xlabel('Verbrauch Käse (kg)')
plt.title('Linear Regression Result')
plt.legend()
plt.show()