# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec

baboon1 = 'venv/data/baboon1.jpg'
baboon2 = 'venv/data/baboon2.jpg'

beach = 'venv/data/beach.jpg'
bear = 'venv/data/bear.jpg'

loup = 'venv/data/loup.png'
johnny = 'venv/data/johnny-hallyday-hommage.jpg'

img1 = cv2.imread(loup)
img2 = cv2.imread(johnny)

height_wolf, width_wolf, channels = img1.shape
height_johnny, width_johnny, channels = img2.shape

resized_wolf = cv2.resize(img1,(width_johnny,height_johnny), interpolation = cv2.INTER_CUBIC)


cv2.imshow('Image 1', resized_wolf)
cv2.imshow('Image 2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

average = cv2.add(img1, img1) /2
cv2.imshow('average',average)

dst = cv2.addWeighted(resized_wolf,0.5,img2,0.5,0)
cv2.imshow('Blended image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_to_save = 'venv/images/johnnywolf.png'
cv2.imwrite(image_to_save, dst)