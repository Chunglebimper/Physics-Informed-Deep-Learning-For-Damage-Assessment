"""
high = 0
line_num = 0
evals = 0
with open("/home/caiden/PycharmProjects/Physics-Informed-Deep-Learning-For-Damage-Assessment/filenameWeights50Trials.txt", "r") as f:
    for line in f.readlines():
        line_num += 1
        if line[0:15] == "Best Macro F1: ":
            print(line, end="")
            evals += 1
            F1 = int(line[-5:-1])
            if F1 > high:
                high = F1

            if F1 == 4418:
                break
        if line[0:34] == "Final class weights used in loss: ":
            print(line, end="")

print(high)
print(line_num)
print(evals)

"""
from PIL import Image
import rasterio
import numpy as np

GROUND_TRUTH = '/home/caiden/PycharmProjects/Physics-Informed-Deep-Learning-For-Damage-Assessment/data/gt_post/mexico-earthquake_00000000_post_disaster_target.png'
NUMBER_OF_CLASSES = 4

with rasterio.open(GROUND_TRUTH) as src:
    image = src.read(1).squeeze()
    image = (255 * (image / NUMBER_OF_CLASSES)).astype('uint8')
    Image.fromarray(image, mode='L').show() # alternatively use .save('name.png')
