import os, sys, shutil, time, pickle, sklearn, cv2
# sys.path.append('from_scratch')
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from util2stage import get_multiscale_windows, detect, resize, predictimage
from nms import lnms
# TODO: training with NO-SIGN image (negative image) large amount
"""
mini zalo data
0: cam nguoc chieu
1: cam dung va do
2: cam re
3: gioi han toc do
4: cam khac
5: nguy hiem
6: hieu lenh
"""
# svm = SGDClassifier(learning_rate='optimal', loss='modified_huber', penalty='l2', alpha=1e-5, max_iter=5000, verbose=False, n_jobs=8, tol=1e-3)
modelstage1 = './models/Stage1-SGD-2-class.sav'
modelstage2 = './models/Stage2-SGD-8-class.sav'
svm1 = pickle.load(open(modelstage1, 'rb'))
svm2 = pickle.load(open(modelstage2, 'rb'))
listOfFiles = [f for f in os.listdir('./img_test/') if os.path.isfile('./img_test/'+f)]
#For single image:
st = time.time()
filename = '1492.png'
image = cv2.imread('./img_test/'+filename)
# new_image, scale, newsize = resize(image, 1000)
# gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
# hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([255, 255, 255]))
# window_list = np.array((get_multiscale_windows(new_image, mask, newsize)))
predicted_image = predictimage(image, svm1, svm2)
print(time.time()-st, filename)
cv2.imwrite('./img_test/result/'+filename.split('.')[0]+'_result1.png',predicted_image)
"""
for filename in listOfFiles:
	# filename = '1502.png'
	image = cv2.imread('./img_test/'+filename)
	new_image, scale, newsize = resize(image, 1000)
	gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
	# mask = cv2.inRange(hsv, np.array([0, 65, 70]), np.array([221, 210, 255]))
	mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([255, 255, 255]))
	window_list = np.array((get_multiscale_windows(new_image, mask, newsize)))
	st = time.time()
	predicted_image = predictimage(new_image, mask, svm1, svm2)
	cv2.imwrite('./img_test/result/'+filename.split('.')[0]+'_result1.png',predicted_image)
	# cv2.imwrite('../img_test/result/'+filename.split('.')[0]+'_mask.png',mask)
"""