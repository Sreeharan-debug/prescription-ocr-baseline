import easyocr
import cv2

# Load image
image = cv2.imread("test.png")

# Initialize reader
reader = easyocr.Reader(['en'])

# Extract text
results = reader.readtext(image)

print("\nDetected Text:\n")

for (bbox, text, prob) in results:
    print(text)