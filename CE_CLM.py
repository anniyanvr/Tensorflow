########################Face detection using Viola Jones Algorithm###########
import numpy as np
import cv2
import copy
from PIL import Image
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('img4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = copy.copy(img)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#for (x,y,w,h) in faces:
 #    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
  #   roi_gray = gray[y:y+h, x:x+w]
   #  roi_color = img[y:y+h, x:x+w]
    # eyes = eye_cascade.detectMultiScale(roi_gray)
     #for (ex,ey,ew,eh) in eyes:
      #   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
 
#cv2.imwrite("face_detected_11.jpg", img)

faces = face_cascade.detectMultiScale(gray2, 1.3, 5)
for (x,y,w,h) in faces:
     cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
     roi_gray = gray[y:y+h, x:x+w]
     roi_color = img[y:y+h, x:x+w]
     eyes = eye_cascade.detectMultiScale(roi_gray)
     for (ex,ey,ew,eh) in eyes:
         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
 
plt.imshow(img2)
plt.axis('off')
plt.show()
#################################################################################

##############################Facial Landmarking using CE-CLM model (Displaying landmarks on the whole face)############
# import the necessary packages
from imutils import face_utils
import numpy as np
import dlib
import cv2
import csv
from PIL import Image
import matplotlib.pyplot as plt

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
#detector = dlib.get_lbprofile_face_detector()
predictor = dlib.shape_predictor(p)

# load the input image, resize it, and convert it to grayscale
img = cv2.imread('face_detected_image.jpg')
#image = imutils.resize(image, width=500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		cv2.circle(img, (x, y), 1, (0, 255, 255), -1)

# show the output image with the face detections + facial landmarks
#cv2.imshow("Output", img)
#cv2.waitKey(0)
plt.imshow(img)
plt.axis('off')
plt.show()

################################################################################

#######################Facial Landmarking using CE-CLM Model (Using ROI)########

#python face_parts_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import csv
from matplotlib import pyplot as plt
import time

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
start = time.time()

def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((77, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 77):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread('face_detected_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)
myFile = open('landmarks.csv', 'w')
# loop over the face detections
for (i, rect) in enumerate(rects):
# determine the facial landmarks for the face region, then
# convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	
	# loop over the face parts individually
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		# clone the original image so we can draw on it, then
		# display the name of the face part on the image
		clone = image.copy()
		cv2.putText(clone, name, (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
                
		# loop over the subset of facial landmarks, drawing the
		# specific face part
		for (x, y) in shape[i:j]:
			cv2.circle(clone, (x, y), 1, (0, 255, 255), -1)

		# extract the ROI of the face region as a separate image
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		roi = image[y:y + h, x:x + w]
		roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

		# show the particular face part
		pts = shape[i:j]
		print(pts,name)
		writer = csv.writer(myFile)
		writer.writerows(pts)
		writer.writerows(str(name))
		#f.write(str(pts))
		#f.write(str(name) + "\n")
		plt.imshow(roi)
		plt.imshow(clone)
		plt.show()
	    	
	# show the output image with the face detections + facial landmarks
	#output = face_utils.visualize_facial_landmarks(image, shape)
	end = time.time()
	print(end - start)
################################################################################

###############################EUCLIDEAN DISTANCE###############################

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
