import tensorflow.keras
import matplotlib.pyplot as plt
import cv2

model_path = 'folder/modelo';
a1 = 'folder/aprint.png';
a2 = 'folder/awpp.jpeg';

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

test_image = cv2.imread(a1)
test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
test_image = test_image / 255
test_image  = test_image.reshape((-1,) + test_image.shape)

print(test_image.dtype)
print(test_image.shape)

y_pred = model.predict_classes(test_image)
print(y_pred)
classname = y_pred[0]
print("Class: ",classname)