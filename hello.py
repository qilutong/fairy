import tensorflow as tf

import fairy




mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

import fairy.tail as ft


# tmp = ft.data_zip((x_train, ft.add_dim(y_train, 1)))
#
# print(type(tmp))
# print(type(tmp[0]))
data = fairy.data.Dataset((x_train, y_train)).repeat().shuffle().batch(32).make_iterator()


#
# print(data.shape)
model.fit_generator(data, epochs=5, steps_per_epoch=60000//32)
# model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)