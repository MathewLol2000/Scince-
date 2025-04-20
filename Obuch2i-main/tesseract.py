import pytesseract
from PIL import Image

image = Image.open('ru_notebooks/images/0_0.jpg')
string = pytesseract.image_to_string(image, lang='rus')

print(string)