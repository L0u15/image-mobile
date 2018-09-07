import cv2
import numpy as np
from matplotlib import pyplot as plt

tests = dict()
lines = []
names = ['beach.jpg', 'dog.jpg', 'polar.jpg', 'bear.jpg', 'lake.jpg', 'moose.jpg']

ref_rgb = cv2.imread('venv/data/waves.jpg')
if ref_rgb is None:
    print("[ERROR] img not found")
    exit()

ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_BGR2GRAY)

hist_ref = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
normalized_hist_ref = cv2.normalize(hist_ref, None)

for name in names:
    img_rgb = cv2.imread('venv/data/%s' % (name))
    if img_rgb is not None:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        normalized_hist = cv2.normalize(hist,None)
        result=cv2.compareHist(normalized_hist_ref, normalized_hist,cv2.HISTCMP_CHISQR)
        print("%s compare to ref: %s"%(name,result))

        line, = plt.plot(normalized_hist, label=name)
        lines.append(line)


line_ref, = plt.plot(normalized_hist, label='ref')
lines.append(line_ref)
plt.legend(handles=lines)
plt.xlim([0, 256])
plt.show()
