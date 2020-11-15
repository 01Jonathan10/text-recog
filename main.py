#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PIL import Image
import pandas as pd
import cv2
from sliding_window import find_text
from text_detect import TextDetector
import sys
	
def main():
	print ("Begin")
	
	load_file = sys.argv[1] == "load" if len(sys.argv) > 1 else None
	
	text_detector = TextDetector(load_file=load_file).trained_model
	
	image_path = "img"
	
	for subdir, dirs, files in os.walk(image_path):
		for file in files:
			filepath = subdir + os.sep + file
			image = cv2.imread(filepath)
			
			find_text(image, text_detector)
			
			break
					
	print(f"Text Extracted from the files in '{image_path}' folder & saved to list..")	

main()
