# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec

baboon1 = 'venv/data/baboon1.jpg'
baboon2 = 'venv/data/baboon2.jpg'

beach = 'venv/data/beach.jpg'
bear = 'venv/data/bear.jpg'

img1 = cv2.imread(beach)
img2 = cv2.imread(bear)
cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

average = cv2.add(img1, img1) /2
cv2.imshow('average',average)

dst = cv2.addWeighted(img1,0.5,img2,0.5,0)
cv2.imshow('Blended image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()