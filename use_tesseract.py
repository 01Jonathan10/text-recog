from PIL import Image
import pytesseract

im = Image.open("mnist/snippets/a.png")

text = pytesseract.image_to_string(im,config='--psm 10')

print(text)