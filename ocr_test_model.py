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
test_path = 'test/p.png'
# test_path = 'test/modelprint.png'
# test_path = 'test/aprint.png'

# test_path2 = 'new_file.png'
model = tensorflow.keras.models.load_model(model_path)

# image = Image.open(test_path)
# if image.mode == 'RGBA':
#     r,g,b,a = image.split()
#     rgb_image = Image.merge('RGB', (r,g,b))

#     inverted_image = PIL.ImageOps.invert(rgb_image)

#     r2,g2,b2 = inverted_image.split()

#     final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))

#     final_transparent_image.save('new_file.png')

# else:
#     inverted_image = PIL.ImageOps.invert(image)
#     inverted_image.save('new_name.png')

labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

test_image = cv2.imread(test_path)
test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image,(28,28)).astype('float32')
test_image = test_image / 255
test_image = test_image.reshape(test_image.shape + (1,))

y_pred = model.predict(np.array([test_image]))
predictions = y_pred[0]
result = np.argmax(predictions)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(labels[np.argmax(predictions)], 100 * np.max(predictions))
)

img = keras.preprocessing.image.load_img(
    test_path, target_size=(28, 28), color_mode='grayscale'
)       # 107 98 target_size=(img_height, img_width)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tensorflow.expand_dims(img_array, 0) # Create a batch
img_array = np.array(img_array[0]).astype('float32')
img_array = img_array / 255

cv2.imshow("Window", img_array)
cv2.waitKey(0)

predictions = model.predict(np.array([img_array]))
score = tensorflow.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(labels[np.argmax(score)], 100 * np.max(score))
)