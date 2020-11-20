import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

model_path = 'test/modelo'
test_path = 'test/modelprint.png'

model = tensorflow.keras.models.load_model(model_path)

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

test_image = cv2.imread(test_path)
test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image,(28,28)).astype('float32')
test_image = test_image / 255
test_image = test_image.reshape(test_image.shape + (1,))

y_pred = model.predict(np.array([test_image]))
predictions = y_pred[0]
result = np.argmax(predictions)
print("Class: ",result)
print(model.summary())

