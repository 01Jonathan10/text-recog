# import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '' # hides the GPU from tensorflow (for science)
# import gzip
import tensorflow.keras
import matplotlib.pyplot as plt
# import numpy as np
# import random
# import struct
# import time

model_path = '/home/seaquest/Arthur/ic/text-recog/folder/modelo'

model_new = tensorflow.keras.models.load_model(model_path)
print(model_new.layers)
results_new = model_new.evaluate(test_X, test_y)
print('Loss: %.2f%%, Accuracy: %.2f%%' % (results_new[0]*100, results_new[1]*100))

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
