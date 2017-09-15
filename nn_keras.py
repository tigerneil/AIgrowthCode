import keras

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


print('training data shape is ', train_images.shape)
print('total training data is ', len(train_labels))

print('test data shape is ', test_images.shape)
print('total test data is ', len(test_labels))

print('test_labels:')
print(test_labels)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(784, )))
model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy']
)

train_images = train_images.reshape((60000, 784))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 784))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test_acc: ', test_acc)
