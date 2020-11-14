import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt
import os
from PIL import Image

image_path = "img/"
ex_txt = []

def traverse(directory):
	path, dirs, files = next(os.walk(directory))
	fol_nm = os.path.split(os.path.dirname(path))[-1]
	print(f'Number of files found in "{fol_nm}" : ',len(files))
	
def TxtExtract(directory):
	"""
	This function will handle the core OCR processing of images.
	"""
	
	for subdir, dirs, files in os.walk(directory):
		for file in files:
			filepath = subdir + os.sep + file
			text = pytesseract.image_to_string(Image.open(filepath), timeout=5)
			if not text:
				ex_txt.extend([[file, "blank"]])
			else:   
				ex_txt.extend([[file, text]])
			break
				
	fol_nm = os.path.split(os.path.dirname(subdir))[-1]
	
	print(f"Text Extracted from the files in '{fol_nm}' folder & saved to list..")	

TxtExtract(image_path)