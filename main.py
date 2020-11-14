#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytesseract
import os
from PIL import Image

image_path = "img/"
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

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
				
			if len(ex_txt) >= 10:
				break
				
	fol_nm = os.path.split(os.path.dirname(subdir))[-1]
	
	print(f"Text Extracted from the files in '{fol_nm}' folder & saved to list..")	

TxtExtract(image_path)

ext_df = pd.DataFrame(ex_txt,columns=['FileName','Text'])
#Inspect the dataframe
ext_df.head()

print(ex_txt)