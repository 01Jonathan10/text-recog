import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.keras 
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import PIL.ImageOps

model_path = 'test/modelo'
test_path = 'test/aprint.png'
test_path2 = 'new_file.png'
model = tensorflow.keras.models.load_model(model_path)

image = Image.open(test_path)
if image.mode == 'RGBA':
    r,g,b,a = image.split()
    rgb_image = Image.merge('RGB', (r,g,b))

    inverted_image = PIL.ImageOps.invert(rgb_image)

    r2,g2,b2 = inverted_image.split()

    final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))

    final_transparent_image.save('new_file.png')

else:
    inverted_image = PIL.ImageOps.invert(image)
    inverted_image.save('new_name.png')

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
labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

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

testtt = test_path2

img = keras.preprocessing.image.load_img(
    testtt, target_size=(28, 28) ,color_mode="grayscale"
)       # 107 98 target_size=(img_height, img_width)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tensorflow.expand_dims(img_array, 0) # Create a batch
print ( img_array )
predictions = model.predict(img_array)
score = tensorflow.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(labels[np.argmax(score)], 100 * np.max(score))
)