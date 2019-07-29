#########################Face Detection using Pretrained cnn####################
from PIL import Image
from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
import dlib
import cv2
import tensorflow as tf
import argparse
import os
import time

def write_to_disk(image, face_cordinates):
    '''
    This function will save the cropped image from original photo on disk 
    '''
    for (x1, y1, w, h) in face_cordinates:
        cropped_face = image[y1:y1 + h, x1:x1 + w]
        cv2.imwrite(str(y1) + ".jpg", cropped_face)

w = "mmod_human_face_detector.dat"
# initialize hog + svm based face detector
hog_face_detector = dlib.get_frontal_face_detector()
# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1(w)
image = cv2.imread('img1.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.

    # Get faces from image
    #start = time.time()
    #faces_hog = hog_face_detector(gray, 1)
    #end = time.time()
    #print("HOG : ", format(end - start, '.2f'))

    # apply face detection (cnn)
start = time.time()
faces_cnn = cnn_face_detector(image, 1)
end = time.time()
print("CNN : ", format(end - start, '.2f'))

    # For each detected face, draw boxes.

    # HOG + SVN
    #for (i, face) in enumerate(faces_hog):
        # Finding points for rectangle to draw on face
     #   x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Drawing simple rectangle around found faces
      #  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # CNN
for (i, face) in enumerate(faces_cnn):
  # Finding points for rectangle to draw on face
  x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
  # Drawing simple rectangle around found faces
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
  # write at the top left corner of the image
  # for color identification
  img_height, img_width = image.shape[:2]
  #cv2.putText(image, "Dlib HOG + SVN", (img_width - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
  #           (0, 0, 255), 2)
  cv2.putText(image, "", (img_width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
              (0, 255, 0), 2)
  # Show the image
  #plot_filepath = os.path.join('/content/drive/My Drive/final_code', "plot.jpg")
  #plt.savefig(plot_filepath)
  plt.imshow(image)
  plt.show()
  #display(image)
  #cv2.imshow('Output', image
  #cv2.waitKey()
  #cv2.destroyAllWindows()
################################################################################

###########################Facial Landmarking using Pretrained CNN##############

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
import dlib
import tensorflow as tf
import cv2
import argparse
import os
from image_utility import save_image, generate_random_color, draw_border
from imutils import face_utils

w = "shape_predictor_68_face_landmarks.dat"
 # landmark predictor
predictor = dlib.shape_predictor(w)
l = "cnn"
d = "mmod_human_face_detector.dat"
face_detector = dlib.cnn_face_detection_model_v1(d)
p = "deploy.prototxt.txt"
m = "res10_300x300_ssd_iter_140000.caffemodel"
face_detector = cv2.dnn.readNetFromCaffe(p, m)
image = cv2.imread('face_cnn.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces_cnn = cnn_face_detector(gray, 1)

# CNN
for (i, face) in enumerate(faces_cnn):
  # Finding points for rectangle to draw on face
  x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
  # Drawing simple rectangle around found faces
  #cv2.rectangle(image, (x, y), (x + w, y + h), generate_random_color(), 2)
  # Make the prediction and transfom it to numpy array
  shape = predictor(gray, face.rect)
  shape = face_utils.shape_to_np(shape)
  # Draw on our image, all the finded cordinate points (x,y)
  for (x, y) in shape:
    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    # Show the image
    #show_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()
    #cv2.imshow('Output', image)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
################################################################################

##################################Euclidean Distance###########################
import numpy as np
import argparse
import imutils
import cv2
import math  

p1 = [191,173]
p2 = [252,187]
p3 = [168,179]
p4 = [268,160]
p5 = [190, 172]
p6 = [252, 171]
p7 = [271, 172]
p8 = [318, 177]
p9 = [208, 185]
p10 = [216, 189]
p11 = [277, 190]
p12 = [288, 192]
p13 = [261, 185]
p14 = [276, 236]
p15 = [168, 179]
p16 = [326, 182]
p17 = [255, 68]
p18 = [259, 110]
p19 = [239, 65]
p20 = [238, 67]
p21 = [150, 160]
p22 = [315, 172]

distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
distance1 = math.sqrt( ((p3[0]-p4[0])**2)+((p3[1]-p4[1])**2) )
distance2 = math.sqrt( ((p5[0]-p6[0])**2)+((p5[1]-p6[1])**2) )
distance3 = math.sqrt( ((p7[0]-p8[0])**2)+((p7[1]-p8[1])**2) )
distance4 = math.sqrt( ((p9[0]-p10[0])**2)+((p9[1]-p10[1])**2) )
distance5 = math.sqrt( ((p11[0]-p12[0])**2)+((p11[1]-p12[1])**2) )
distance6 = math.sqrt( ((p13[0]-p14[0])**2)+((p13[1]-p14[1])**2) )
distance7 = math.sqrt( ((p15[0]-p16[0])**2)+((p15[1]-p16[1])**2) )
distance8 = math.sqrt( ((p17[0]-p18[0])**2)+((p17[1]-p18[1])**2) )
distance9 = math.sqrt( ((p19[0]-p20[0])**2)+((p19[1]-p20[1])**2) )
distance10 = math.sqrt( ((p21[0]-p22[0])**2)+((p21[1]-p22[1])**2) )
name="Forehead distance (Tr-G):"
name1="Left and Right eyes (ex-ex):"
name2="Left and Right eyes (en-en):"
name3="Left eye(ex-en):"
name4="Right eye(ex-en):"
name5="Right Eyebrow Distance:"
name6="Left Eyebrow Distance:"
name7="Upper cheek(zy-zy):"
name8="Nose Distance:"
name9=" Mouth Distance:"
name10="Jawline Distance:"
print(str(name),distance)
print(str(name1),distance1)
print(str(name2),distance2)
print(str(name3),distance3)
print(str(name4),distance4)
print(str(name5),distance5)
print(str(name6),distance6)
print(str(name7),distance7)
print(str(name8),distance8)
print(str(name9),distance9)
print(str(name10),distance10)
################################################################################
