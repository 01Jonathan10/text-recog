#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import imutils
import cv2
import tensorflow.keras
import cv2
import numpy as np
import time
from sliding_window import sliding_window

(winW, winH) = (28, 28)

snippet_path = "mnist/snippets/a.png"

def find_text(image):
	result = ''
	model = tensorflow.keras.models.load_model('test/modelo')
	image = imutils.resize(image, height=28)
	labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
	for (x, y, window) in sliding_window(image, stepSize=5, windowSize=(winW, winH)):
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		data = get_grayscale_sector(window)
		predictions = model.predict(np.array([data]))[0]
		guess = np.argmax(predictions) if np.max(predictions) > 0.3 else None
		letter = labels[guess] if guess else "-"
				
		clone = image.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)
		cv2.waitKey(1)
		time.sleep(0.025)
		
		result = result + letter
		
	print (result)
	return result
			
def get_grayscale_sector(image):
	sector = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype('float32')
	sector = sector / 255
	sector = sector.reshape(sector.shape + (1,))
	return sector
	
	
find_text(cv2.imread(snippet_path))
