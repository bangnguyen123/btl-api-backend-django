from rest_framework import serializers

from rest_framework.serializers import (
      ModelSerializer,
)
from django.core.files import File
from image_app.models import MyImage
import cv2
import numpy as np
import os, sys, shutil, time, pickle, sklearn, cv2
sys.path.append('hog_svm_detection')
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from util2stage import get_multiscale_windows, detect, resize, predictimage
from nms import lnms

# svm = SGDClassifier(learning_rate='optimal', loss='modified_huber', penalty='l2', alpha=1e-5, max_iter=5000, verbose=False, n_jobs=8, tol=1e-3)
modelstage1 = './models/Stage1-SGD-2-class.sav'
modelstage2 = './models/Stage2-SGD-8-class.sav'
svm1 = pickle.load(open(modelstage1, 'rb'))
svm2 = pickle.load(open(modelstage2, 'rb'))

class imageSerializer(ModelSerializer):

   def create(self, validated_data):
      image = validated_data['model_pic'].read()
      img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
      # img_np = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      print("bang")
      #
      img_gray, x = predictimage(img, svm1, svm2)
      #
      # img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
      cv2.imwrite("image_app/image_after_predict/bang.jpg",img_gray)
      pic = MyImage()
      pic.model_pic.save(validated_data['model_pic'].name, File(open("image_app/image_after_predict/bang.jpg",'rb')))
      return pic

   class Meta:
      model = MyImage
      fields = [
         'model_pic'
      ]
