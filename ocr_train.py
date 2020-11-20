import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '' # hides the GPU from tensorflow (for science)
import gzip
import tensorflow.keras
import matplotlib.pyplot as plt
import numpy as np
import random
import struct
import time

#para baixar
#List computing devices available to tensorflow:
from tensorflow.python.client import device_lib
device_list = device_lib.list_local_devices()
# [x.name for x in device_list]
# %matplotlib inline
random.seed(12345)
image_dir = 'mnist'
labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
categories = len(labels)
batch_size = 1024
epochs = 50
model_path = 'test/modelo'

#codigo copiado p/ abrir os index do emnist
def read_idx(filename):
	print('Processing data from %s.' % filename)
	with gzip.open(filename, 'rb') as f:
		z, dtype, dim = struct.unpack('>HBB', f.read(4))
		print('Dimensions:', dim)
		shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dim))
		print('Shape:', shape)
		return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

#codigo copiado p/ carregar o emnist
def load_emnist():
	train_images = os.path.join(image_dir, 'emnist-mnist-train-images-idx3-ubyte.gz')
	train_labels = os.path.join(image_dir, 'emnist-mnist-train-labels-idx1-ubyte.gz')
	test_images = os.path.join(image_dir, 'emnist-mnist-test-images-idx3-ubyte.gz')
	test_labels = os.path.join(image_dir, 'emnist-mnist-test-labels-idx1-ubyte.gz')

	train_X = read_idx(train_images)
	train_y = read_idx(train_labels)
	test_X = read_idx(test_images)	
	test_y = read_idx(test_labels)
	return (train_X, train_y, test_X, test_y)

raw_train_X, raw_train_y, raw_test_X, raw_test_y = load_emnist()

#avaliando data set
print("shapes",raw_train_X.shape, raw_train_y.shape, raw_test_X.shape, raw_test_y.shape)

# #mostrando imagem que temos no dataset
# i = random.randint(0, raw_train_X.shape[0])
# fig, ax = plt.subplots()
# ax.clear()
# ax.imshow(raw_train_X[i].T, cmap='gray')
# title = 'label = %d = %s' % (raw_train_y[i], labels[raw_train_y[i]])
# ax.set_title(title, fontsize=20)
# plt.show()

train_X = raw_train_X.astype('float32')
test_X = raw_test_X.astype('float32')
train_X /= 255
test_X /= 255

train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
test_X = test_X.reshape(test_X.shape[0], 28, 28, 1)

train_y = tensorflow.keras.utils.to_categorical(raw_train_y)
test_y = tensorflow.keras.utils.to_categorical(raw_test_y)
# import tensorflow
#MONTANDO A REDE CONVOLUCIONAL
model = tensorflow.keras.models.Sequential()

model.add(tensorflow.keras.layers.Conv2D(32,
			 kernel_size=(5, 5),
			 strides=(2, 2),
			 input_shape=(28, 28, 1),
			 activation='relu'))

model.add(tensorflow.keras.layers.Conv2D(64,
			 kernel_size=(3, 3),
			 activation='relu'))

model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tensorflow.keras.layers.Dropout(0.25))

model.add(tensorflow.keras.layers.Flatten())

model.add(tensorflow.keras.layers.Dense(128, activation='relu'))

model.add(tensorflow.keras.layers.Dropout(0.25))

model.add(tensorflow.keras.layers.Dense(categories, activation='softmax'))

# compilar
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

# treinar
t1 = time.time()
fit = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_X, test_y))
t2 = time.time()
print('Elapsed time: %ds' % (t2 - t1))

# testar
results = model.evaluate(test_X, test_y)
# Showing the loss and accuracy results:
print(results[0]*100, results[1]*100)

#ploting o modelo
plt.figure(figsize=(12, 6), dpi=96)
plt.subplot(1, 2, 1)
plt.plot(fit.history['loss'])
plt.plot(fit.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(1, 2, 2)
plt.plot(fit.history['accuracy'])
plt.plot(fit.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.show()

#salvar modelo
model.save(model_path)

#carregar modelo e testar denovo
model_new = tensorflow.keras.models.load_model(model_path)
print(model_new.layers)
results_new = model_new.evaluate(test_X, test_y)
print('Loss: %.2f%%, Accuracy: %.2f%%' % (results_new[0]*100, results_new[1]*100))

