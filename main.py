#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PIL import Image
import pandas as pd
import cv2
from sliding_window import find_text
from text_detect import TextDetector
import sys
import numpy as np
	
def main():
	print ("Begin")
	
	load_file = sys.argv[1] == "load" if len(sys.argv) > 1 else None
	
	text_detector = TextDetector(load_file=load_file).trained_model
	
	image_path = "img"
	
	for subdir, dirs, files in os.walk(image_path):
		for file in files:
			filepath = subdir + os.sep + file
			image = cv2.imread(filepath)
			
			processed = np.copy(image)
			filtered_img = np.empty(image.shape, dtype=np.uint8)
			
			text_locations = find_text(processed, text_detector)
			
			for location, scale in text_locations:
				x = int(location[0])
				y = int(location[1])
				filtered_img = add_rect(x, y, int(40/scale), int(40/scale), 0.2, filtered_img)
						
			res = cv2.addWeighted(image, 1, filtered_img, 0.7, 1.0)
			cv2.imshow('image', filtered_img)
			cv2.waitKey(0)
			
			break
					
	print(f"Text Extracted from the files in '{image_path}' folder & saved to list..")	
	

def show_contours(im):
	im = np.copy(im)
	im2 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(im2,127,255,0)
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	for i in range(0, len(contours)):
		if (i % 2 == 0):
		   cnt = contours[i]
		   x,y,w,h = cv2.boundingRect(cnt)
		   im = cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
		   
	cv2.imshow('image', im)
	cv2.waitKey(0)
	

def add_rect(x, y, w, h, alpha, img):
	sub_img = img[y:y+h, x:x+w]
	white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
	res = cv2.addWeighted(sub_img, 1-alpha, white_rect, alpha, 1.0)
	img[y:y+h, x:x+w] = res
	
	return img

main()
