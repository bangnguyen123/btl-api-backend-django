import os, sys, shutil
import numpy as np
import cv2

from skimage.feature import hog
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from imutils import paths
import argparse
import time
import pickle
import random
from tqdm import tqdm

def extract_feature(path):
	print("[INFO] extracting training features from {}...".format(path))
	data = []
	labels = []
	filenames = []
	index = 0
	for imagePath in tqdm(paths.list_images(path)):
		index +=1
		make = imagePath.split("\\")[-2]
	
		# load the image, convert it to grayscale, and detect edges
		image = cv2.imread(imagePath)
		try:			
			gray = cv2.resize(image, (96, 96))
			gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
			# extract Histogram of Oriented Gradients from the logo
			hogFeature = hog(gray,orientations=9,pixels_per_cell=(8, 8),cells_per_block=(2, 2),transform_sqrt=True,visualize=False,block_norm='L2')
			data.append(hogFeature)
			labels.append(make)
			filenames.append(imagePath)
		except:
			print(imagePath)

	data = np.stack(data, axis=0)
	labels = np.stack(labels, axis=0)
	print("[INFO] Feature shape: {}".format(data.shape))
	return data, labels, filenames

def main(stage=1):
	if stage == 1:
		trainPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\stage1_classifier\train'
		valPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\stage1_classifier\valid'
		modelPath = 'models/Stage1-SGD-2-class.sav'
	else: 
		trainPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\stage2_classifier\train'
		valPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\stage2_classifier\valid'
		modelPath = 'models/Stage2-SGD-8-class.sav'

	# construct the argument parse and parse command line arguments

	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--training", required=False, default=trainPath, help="Path to the training dataset")
	ap.add_argument("-v", "--validation", required=False, default=valPath, help="Path to the validation dataset")
	args = vars(ap.parse_args())
	
	# initialize the data matrix and labels
	start = time.time()
	data_train, labels_train, _ = extract_feature(path=args["training"])
	data_val, labels_val, filenames_val = extract_feature(path = args["validation"])

	print("[INFO] Finish extracting HoG features. Total time: {}".format(time.time()-start))
	# define classifier

	start = time.time()
	print("[INFO] Training...")
	clf = SGDClassifier(learning_rate='optimal', loss='hinge', penalty='l2', alpha=0.001, max_iter=15000, verbose=False, n_jobs=-1, tol=1e-3, early_stopping=True)
 
	# calibration for probability estimation
	clf_with_prob = CalibratedClassifierCV(clf)
	clf_with_prob.fit(data_train, labels_train)
	print("[RESULT] Training accuracy:", clf_with_prob.score(data_train, labels_train))
	print("[INFO] Finish training SVM model. Total time: {}".format(time.time()-start))
	# "train" the nearest neighbors classifier

	print("[INFO] Saving model...")
	pickle.dump(clf_with_prob, open(modelPath, 'wb'))  

	print('[INFO] Validation accuracy:', clf_with_prob.score(data_val, labels_val))
	# print('Test accuracy on Scratch HoG extractor', model2.score(data2, labels))
	print("[RESULT] Confusion matrix...")
	print(metrics.confusion_matrix(clf_with_prob.predict(data_val), labels_val))

if __name__=='__main__':
    
    main(1)

	# probability estimation reference: https://mmuratarat.github.io/2019-10-12/probabilistic-output-of-svm#:~:text=SVMs%20don't%20output%20probabilities,the%20output%20to%20class%20probabilities.&text=One%20standard%20way%20to%20obtain,in%20many%20decent%20SVM%20implementations.
	# Best estimator found by grid search:
	# SGDClassifier(alpha=0.001, max_iter=15000, n_jobs=8, verbose=False)