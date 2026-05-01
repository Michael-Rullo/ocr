from PIL import Image
import numpy as np

def load_image(path):
    try:
        image = Image.open(path)
        gray_image = image.convert('L')
        # shows the image grayscale converted for testing
        # gray_image.show()
        gray_image = np.array(gray_image, dtype=np.uint8)
        return gray_image
    except Exception as e:
        print(f"error loading image: {e}")
        return None