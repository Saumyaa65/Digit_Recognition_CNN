import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, accuracy_score

(x_train, y_train), (x_test, y_test)= mnist.load_data()
# x_train.shape= (60000, 28, 28), y_train.shape= (60000,)
# x_test.shape= (10000, 28, 28) , y_test.shape= (10000,)

x_train= x_train/255.0
x_test= x_test/255.0

# change shape as original is 2d but cnn need 3d
x_train=x_train.reshape(60000, 28,28,1)
x_test=x_test.reshape(10000, 28,28,1)

input_shape=x_train[0].shape

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),
                                 input_shape=input_shape, activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),
                                 activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

history= model.fit(x_train, y_train, batch_size=128, epochs=10,
          validation_data=(x_test, y_test))

y_pred=np.argmax(model.predict(x_test),axis=-1)
print(y_pred[3], y_test[3])
print(y_pred[3524], y_test[3524])
print(y_pred[123], y_test[123])
print(y_pred[300], y_test[300])
print(y_pred[-1], y_test[-1])

cm=confusion_matrix(y_test, y_pred)
print(cm)
acc_cm=accuracy_score(y_test, y_pred)
print(acc_cm)

# training vs validation accuracy
epoch_range=range(1, 11)
plt.plot(epoch_range, history.history['sparse_categorical_accuracy'])
plt.plot(epoch_range, history.history['val_sparse_categorical_accuracy'])
plt.title("Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc='upper left')
plt.show()

# training vs validation loss
epoch_range=range(1, 11)
plt.plot(epoch_range, history.history['loss'])
plt.plot(epoch_range, history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc='upper left')
plt.show()