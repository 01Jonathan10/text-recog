#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PIL import Image
import pandas as pd
import cv2
from sliding_window import find_text
	
def main():
	image_path = "img"
	for subdir, dirs, files in os.walk(image_path):
		for file in files:
			filepath = subdir + os.sep + file
			# do stuff with file
			image = cv2.imread(filepath)
			
			print (filepath)
			find_text(image)
			
			break
					
	print(f"Text Extracted from the files in '{image_path}' folder & saved to list..")	


main()
