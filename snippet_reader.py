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
import pytesseract
from PIL import Image

(winW, winH) = (28, 28)

snippet_path = "mnist/snippets/a.png"

def find_text(image):
	result = ''
	
	image = imutils.resize(image, height=28)
	for (x, y, window) in sliding_window(image, stepSize=10, windowSize=(winW, winH)):
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		a=27
		img = Image.fromarray(window)
		img.save('teste.png')
		letter = pytesseract.image_to_string(Image.open("teste.png"),config='--psm 10')
		letter2= letter
		letter = letter.rstrip("\n")
		window2 = window
		threshould= 22
		threshoulddiff= 6
		if len(letter) > 3:
			a=27
			while len(letter) > 3 and a > threshould:
				a=a-1
				img = Image.fromarray(window2[:,0:a])
				img.save('teste.png')
				letter = pytesseract.image_to_string(Image.open("teste.png"),config='--psm 10')
				letter3 = letter.rstrip("\n")
				# print ( a )
				# print( len(letter) )
				# print( letter3[0:len(letter)-2] )
				# print( letter2 )
		if len(letter) > 3:
			a=27
			while len(letter) > 3 and a > threshould:
				a=a-1
				img = Image.fromarray(window2[:,27-a:27])
				img.save('teste.png')
				letter = pytesseract.image_to_string(Image.open("teste.png"),config='--psm 10')
				letter3 = letter.rstrip("\n")
				# print ( a )
				# print( len(letter) )
				# print( letter3[0:len(letter)-2] )
				# print( letter2 )
		if len(letter) > 3:
			a=27
			while len(letter) > 3 and a > threshould:
				a=a-1
				img = Image.fromarray(window2[:,27-a:a])
				img.save('teste.png')
				letter = pytesseract.image_to_string(Image.open("teste.png"),config='--psm 10')
				letter3 = letter.rstrip("\n")
				# print ( a )
				# print( len(letter) )
				# print( letter3[0:len(letter)-2] )
				# print( letter2 )

		if len(letter) > 3:
			a=27
			while len(letter) > 3 and a > threshould - threshoulddiff:
				a=a-2
				img = Image.fromarray(window2[:,0:a])
				img.save('teste.png')
				letter = pytesseract.image_to_string(Image.open("teste.png"),config='--psm 10')
				letter3 = letter.rstrip("\n")
				# print ( a )
				# print( len(letter) )
				# print( letter3[0:len(letter)-2] )
				# print( letter2 )
		if len(letter) > 3:
			a=27
			while len(letter) > 3 and a > threshould - threshoulddiff:
				a=a-2
				img = Image.fromarray(window2[:,27-a:27])
				img.save('teste.png')
				letter = pytesseract.image_to_string(Image.open("teste.png"),config='--psm 10')
				letter3 = letter.rstrip("\n")
				# print ( a )
				# print( len(letter) )
				# print( letter3[0:len(letter)-2] )
				# print( letter2 )
		if len(letter) > 3:
			a=27
			while len(letter) > 3 and a > threshould - threshoulddiff:
				a=a-2
				img = Image.fromarray(window2[:,27-a:a])
				img.save('teste.png')
				letter = pytesseract.image_to_string(Image.open("teste.png"),config='--psm 10')
				letter3 = letter.rstrip("\n")
				# print ( a )
				# print( len(letter) )
				# print( letter3[0:len(letter)-2] )
				# print( letter2 )
		if len(letter) == 3 :
			result = result + letter3[0:len(letter)-2]
			# print ( a )
			# print( letter )
	print( "threshould: " + str(threshould) + " threshoulddiff: " + str(threshoulddiff) )
	print (result)
	return result


def find_text_list_append(image):
	result = list()	
	image = imutils.resize(image, height=28)
	for (x, y, window) in sliding_window(image, stepSize=5, windowSize=(winW, winH)):
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		a=27
		letter = list()
		img = Image.fromarray(window)
		img.save('teste.png')
		letter.append( pytesseract.image_to_string(Image.open("teste.png"),config='--psm 10'))
		window2 = window

		while len(letter) > 1 and a > 20:
			a=a-1
			img = Image.fromarray(window2[:,0:a])
			img.save('teste.png')
			letter.append( pytesseract.image_to_string(Image.open("teste.png"),config='--psm 10'))

		a=27
		while len(letter) > 1 and a > 15:
			a=a-1
			img = Image.fromarray(window2[:,27-a:27])
			img.save('teste.png')
			letter.append( pytesseract.image_to_string(Image.open("teste.png"),config='--psm 10'))
		a=27
		while len(letter) > 1 and a > 5:
			a=a-1
			img = Image.fromarray(window2[:,27-a:a])
			img.save('teste.png')
			letter.append( pytesseract.image_to_string(Image.open("teste.png"),config='--psm 10'))
		print(letter)
		result.append(letter)	


	print (result)
	return result

def find_text2(image):
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
