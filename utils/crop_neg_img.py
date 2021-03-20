import cv2
import os
import shutil
import json
import sys
import matplotlib.pyplot as plt
import random

ORIGSIZE = [512,1024,800,1000]
TARGETSIZE = 128

targetPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\negative'
if not os.path.isdir(targetPath):
    os.mkdir(targetPath)

#YOLO format
imgPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_TGM\all_renamed_img'
annotPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_TGM\annotations'
counter = 0
targetNum = 4000

for origSize in ORIGSIZE:
    for index, imgName in enumerate(os.listdir(imgPath)):
        for numCrop in range(3):
            # skip = False
            labelPath = os.path.join(annotPath, imgName[:-4] + '.txt')
            if not os.path.isfile(labelPath):
                continue
            imagePath = os.path.join(imgPath, imgName)
            img = cv2.imread(imagePath)
            height, width, _ = img.shape
            with open(labelPath) as annotFile:
                bboxes = annotFile.readlines()
                for box in bboxes:
                    centerX = random.randint(origSize//2, img.shape[1]-origSize//2)
                    centerY = random.randint(origSize//2, img.shape[0]-origSize//2)
                    x1 = centerX - origSize//2
                    x2 = centerX + origSize//2
                    y1 = centerY - origSize//2
                    y2 = centerY + origSize//2
                    
                    skip = False
                    x, y, w, h = box.split()[1:5]
                    x = float(x) * width
                    y = float(y) * height
                    w = float(w) * width
                    h = float(h) * height
                    # print(x, y, w, h)
                    if (x2>(x-w/2) and y2>(y-h/2)) or (x2>(x-w/2) and y1<(y+h/2)) or (x1<(x+2/2) and y2>(y-h/2)) or (x2<(x+w/2) and y2<(y+h/2)):
                        skip = True
                        # print('skip ne ')
                    if skip:
                        # print('crop ne')
                        cropImg = img[y1:y2,x1:x2,:]
                        cv2.imwrite(os.path.join(targetPath, str(counter+10000) + '.png'), cropImg)
                        counter += 1
                        if counter == targetNum:
                            quit()
                # if index == 2:
                #     quit()
        if index % 1000 ==0 and not index == 0:
            break
