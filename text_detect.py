#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import glob
import numpy as np
from skimage.feature import hog
from sklearn import preprocessing
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report
import os
import pickle

text_path = "./text_nontext-dataset/text"
non_text_path = "./text_nontext-dataset/nontext"

class TextDetector():

	def __init__(self, load_file=False):	
		print ("Init Text Detector")
		
		print ("\tReading Training Data")
		train_text = self.extractfeat(text_path)
		train_nontext = self.extractfeat(non_text_path)
		trainfeat = np.vstack((train_text,train_nontext))
		trainfeat = preprocessing.normalize(trainfeat)
		
		trainlabeltext = np.ones(len(os.listdir(text_path)))
		trainlabelnontext = np.zeros(len(os.listdir(non_text_path )))
		labeltrain = np.hstack((trainlabeltext,trainlabelnontext))
		
		if load_file:
			svclassifier = pickle.loads(open("text_detect_model.txt", "rb").read())
		
		else:
			print("Training Model...")
			svclassifier = SVC(kernel='rbf')	
			svclassifier.fit(trainfeat, labeltrain)
			
			saved = pickle.dumps(svclassifier)
			with open("text_detect_model.txt", "wb") as file:
				file.write(saved)
				file.close()
				
		y_pred_train = svclassifier.predict(trainfeat)
		print("Training Result:\n"+classification_report(labeltrain, y_pred_train))
		
		self.trained_model = svclassifier
		
	@staticmethod
	def extract_from_image(img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (40, 40))
		fd = hog(img, orientations=9, pixels_per_cell=(30, 30), cells_per_block=(1, 1))

		return np.array([fd], 'float64')
	
	@staticmethod
	def preprocess(path):
		img = cv2.imread(path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(img, (40, 40))
		return image

	@staticmethod
	def extractfeat(path):
		list_hog_fd = []
		for files in os.listdir(path):
			img = TextDetector.preprocess(path + "/" + files)
			print("\t" + path + "/" + files)
			fd = hog(img, orientations=9, pixels_per_cell=(30, 30), cells_per_block=(1, 1))
			list_hog_fd.append(fd)
		hog_features = np.array(list_hog_fd, 'float64')
		return hog_features
