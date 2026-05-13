from PIL import Image
import numpy as np

def load_image(path):

    try:
        image = Image.open(path)
    except Exception as e:
        print(f"error loading image: {e}")
        return None
    
    image = np.array(image, dtype=np.uint8)
    gray_image = grayscale(image)
    gray_image = np.array(gray_image, dtype=np.uint8)
    return gray_image
    
def grayscale(image):

    new_image = [[0 for _ in range(len(image[0]))] for _ in range(len(image))]
    
    for i in range(len(image)):
        for j in range(len(image[0])):
            # new_image[i][j] = round(0.2126 * image[i][j][0] + 0.7152 * image[i][j][1] + 0.0722 * image[i][j][2])
            new_image[i][j] = round(0.299 * image[i][j][0] + 0.587 * image[i][j][1] + 0.114 * image[i][j][2])
    return new_image
