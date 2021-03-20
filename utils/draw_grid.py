import matplotlib
import numpy as np
import cv2

if __name__ == '__main__':
        
    im = cv2.imread(r'images\test-3-classes\2\321.png')
    im = cv2.resize(im, (96,96))
    im_ = im.copy()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # cell split for  HOG
    pixels_per_cell=(8, 8)
    cells_per_block=(4, 4)

    for i in range(96//pixels_per_cell[0]):
        for j in range(96//pixels_per_cell[1]):
            if not (i==0 or j==0):
                continue
            cv2.line(im_, (i*pixels_per_cell[0], j*pixels_per_cell[1]), ((i*pixels_per_cell[0]+96, j*pixels_per_cell[1])), color=(0,255,0), thickness=1)
            cv2.line(im_, (i*pixels_per_cell[0], j*pixels_per_cell[1]), ((i*pixels_per_cell[0], j*pixels_per_cell[1]+96)), color=(0,255,0), thickness=1)
            
            
                
            
    cv2.line(im_, (95,0), (95,95), (0,255,0), 1)
    cv2.line(im_, (0,95), (95,95), (0,255,0), 1)
    i=j=2
    cv2.rectangle(im_, (0+32,0),(32+32,32), (255,0,0), 1)
 
    cv2.imwrite('output-images\lines.png', im_)

    
