import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import time
class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time > 300:
            self.model.stop_training = True

mnist = tf.keras.datasets.mnist
(x_train, y_train_origin), (x_test, y_test_origin) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0           # 0과 1 사이의 값으로 변환
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

nb_classes = 10
y_train = keras.utils.to_categorical(y_train_origin, num_classes = nb_classes)  # one-hot encoding
y_test = keras.utils.to_categorical(y_test_origin, num_classes = nb_classes)    # one-hot encoding

print(x_train.shape)
print(y_train.shape)
print(y_test_origin)
print(y_test)

model = keras.Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = keras.optimizers.SGD(learning_rate = 0.01), loss = 'mse', metrics = ['categorical_accuracy'])
# model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01), loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=10, validation_data=(x_test, y_test), callbacks=[MyCallback()])
model.evaluate(x_test, y_test)


score = model.evaluate(x_test, y_test, verbose=0)



print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('mnist_cnn.h5')
print("Saving the model as mnist_cnn.h5")
