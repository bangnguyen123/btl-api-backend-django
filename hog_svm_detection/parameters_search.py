import os, sys, shutil
sys.path.append('from_scratch')
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import cv2
from tqdm import tqdm
from hogExtractor import hogDescriptorScratch, hogDescriptorSkimage
from sklearn.linear_model import SGDClassifier

import numpy as np
from imutils import paths
import argparse
import time

# construct the argument parse and parse command line arguments
trainPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\stage1_classifier\train'
testPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\stage1_classifier\valid'

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=False, default=trainPath, help="Path to the logos training dataset")
ap.add_argument("-t", "--test", required=False, default=testPath, help="Path to the test dataset")
args = vars(ap.parse_args())
 
# initialize the data matrix and labels
print("[INFO] extracting features...")
data1 = []
data2 = []
labels = []

for imagePath in tqdm(paths.list_images(args["training"])):

	make = imagePath.split("\\")[-2]
 
	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	try: 
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.resize(gray, (128, 128))
	
		# extract Histogram of Oriented Gradients from the logo
		hogFeatureSkimage = hogDescriptorSkimage(gray, 
												orientations=9, 
												pixels_per_cell=(8, 8),
												cells_per_block=(4, 4), 
												transform_sqrt=True, 
												visualize=False,
												block_norm='L2')
		# hogFeatureScratch = hogDescriptorScratch(gray, 
		# 										orientations=9, 
		# 										cell_size=(8,8), 
		# 										cells_per_block=(2,2),
		# 										block_norm='L2',
		# 										visualize=False)
		# update the data and labels
		# print(H.shape)
		data1.append(hogFeatureSkimage)
		# data2.append(hogFeatureScratch)
		labels.append(make)
	except:
		print(imagePath)

data1 = np.stack(data1, axis=0)
# data2 = np.stack(data2, axis=0)
labels = np.stack(labels, axis=0)


# param_grid = {'C': [1.0, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
#               'kernel': ['rbf', 'linear', 'poly' ]}
param_grid = {'alpha': [1e-8, 1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5],
              'penalty': ['l2', 'l1']}
# SVC = svm.SVC(kernel='rbf', class_weight='balanced')
clf = SGDClassifier(learning_rate='optimal', loss='hinge', penalty='l2', alpha=1e-5, max_iter=15000, verbose=False, n_jobs=4, tol=1e-3)

clf = GridSearchCV(clf, param_grid, cv=5, n_jobs=4, verbose=1)
clf = clf.fit(data1, labels)
# print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
print(clf.cv_results_)

"""
Best estimator found by grid search:
SVC(C=10.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""
