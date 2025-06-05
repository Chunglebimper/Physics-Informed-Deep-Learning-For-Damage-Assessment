from PIL import Image
import numpy as np

def testDisplay(image_array):
    image_array = (255 * (image_array / image_array.max())).astype(np.uint8)

    composite_image = Image.fromarray(image_array, mode='RGB')
    composite_image.save('./processed/composite_rgb.png')