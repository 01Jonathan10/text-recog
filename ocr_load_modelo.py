# import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '' # hides the GPU from tensorflow (for science)
# import gzip
import opencv_python
import tensorflow.keras
import matplotlib.pyplot as plt
# import numpy as np
# import random
# import struct
# import time

model_path = '/home/seaquest/Arthur/ic/text-recog/folder/modelo';
a1 = '/home/seaquest/Arthur/ic/text-recog/folder/aprint.png';
a2 = '/home/seaquest/Arthur/ic/text-recog/folder/awpp.jpeg';

model = tensorflow.keras.models.load_model(model_path)
print(model.layers)
# results_new = model_new.evaluate(test_X, test_y)
# print('Loss: %.2f%%, Accuracy: %.2f%%' % (results_new[0]*100, results_new[1]*100))

# #ploting o modelo
# plt.figure(figsize=(12, 6), dpi=96)
# plt.subplot(1, 2, 1)
# plt.plot(fit.history['loss'])
# plt.plot(fit.history['val_loss'])
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(['train', 'test'], loc='upper left')
# plt.subplot(1, 2, 2)
# plt.plot(fit.history['accuracy'])
# plt.plot(fit.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(['train', 'test'], loc='upper left')
# plt.tight_layout()
# plt.show()


test_image = image.load_img(a1,color_mode="grayscale",target_size=(28,28,1))
print(test_image.format)
print(test_image.mode)
print(test_image.size)

test_image = image.img_to_array(test_image)
test_image = test_image / 255
test_image  = test_image.reshape((-1,) + test_image.shape)

print(test_image.dtype)
print(test_image.shape)

y_pred = model.predict_classes(test_image)
print(y_pred)
classname = y_pred[0]
print("Class: ",classname)