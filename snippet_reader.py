#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import imutils
import cv2
import tensorflow.keras
import cv2
import numpy as np
from sliding_window import sliding_window

(winW, winH) = (28, 28)

snippet_path = "mnist/snippets/a.png"

def find_text(image):
	result = []
	model = tensorflow.keras.models.load_model('test/modelo')
	image = imutils.resize(image, height=28)
	for (x, y, window) in sliding_window(image, stepSize=10, windowSize=(winW, winH)):
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		data = get_grayscale_sector(window)
		predictions = model.predict(np.array([data]))[0]
		result = np.argmax(predictions) if np.max(predictions) > 0.5 else None
		print("Class: ",result)
		
	
	return result
			
def get_grayscale_sector(image):
	sector = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype('float32')
	sector = sector / 255
	sector = sector.reshape(sector.shape + (1,))
	return sector
	
	
find_text(cv2.imread(snippet_path))
