# predict using SVM SGD from scratch model
import os, sys, shutil
import numpy as np
import cv2

from skimage.feature import hog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from imutils import paths
import argparse
import time
import pickle

img_path = r'images\test-3-classes\0\692.png'

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--imgPath", required=False, default=img_path, help="Path to test image")
args = vars(ap.parse_args())
 
# initialize the data matrix and labels
print("[INFO] Extracting features...")

imagePath = args["imgPath"]
 
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (96, 96))
 
# extract Histogram of Oriented Gradients from the logo
hogFeature = hog(gray,orientations=9,pixels_per_cell=(8, 8),cells_per_block=(4, 4),transform_sqrt=True,visualize=False,block_norm='L2')

data = np.stack(hogFeature, axis=0)
data = np.expand_dims(data, axis=0)
print('[INFO] Feature size: ', data.shape)

print("[INFO] Loading classifier...")

clf = SGDClassifier(learning_rate='optimal', loss='hinge', penalty='l2', alpha=0.001, max_iter=15000, verbose=False, n_jobs=-1, tol=1e-3, early_stopping=True)
clf = CalibratedClassifierCV(clf)
model_path = '.\models\Stage1-SGD-2-class.sav'

clf = pickle.load(open(model_path, 'rb'))

start = time.time()

label = clf.predict(data.reshape((1,-1)))
score = clf.predict_proba(data.reshape((1,-1)))

labels = ['cam nguoc chieu', 'cam dung va do', 'cam re', 'gioi han toc do', 'cam khac', 'nguy hiem', 'hieu lenh']
print("[INFO] Predicting time: ", time.time()-start)
print('[INFO] Predicted label:', labels[int(label)])
print('[INFO] Confident score: ', score)

"""
mini zalo data
0: cam nguoc chieu
1: cam dung va do
2: cam re
3: gioi han toc do
4: cam khac
5: nguy hiem
6: hieu lenh
7: no sign
"""