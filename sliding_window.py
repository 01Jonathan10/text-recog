#!/usr/bin/env python
# -*- coding: utf-8 -*-

import imutils
import argparse
import time
import cv2
from text_detect import TextDetector

(winW, winH) = (40, 40)

def find_text(image, text_detector):
	for resized in pyramid(image, scale=1.5):
		for (x, y, window) in sliding_window(resized, stepSize=20, windowSize=(winW, winH)):
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			clone = resized.copy()
			region = resized[y:y+winW, x:x+winW]
			data = TextDetector.extract_from_image(region)
			
			result = text_detector.predict(data)[0]

def pyramid(image, scale=1.5, minSize=(30, 30)):
	yield image
	while True:
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		yield image
		
def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])