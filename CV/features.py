import cv2
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
image = img.imread('chinese-1.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
plt.plot(range(0,8), hist[0,0,:])
plt.show()
cv2.normalize(hist, hist)