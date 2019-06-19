import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import utils.bbops as bbops
import utils.helper as helper
import utils.preprocess as preprocess

import argparse

# ArgParse

parser = argparse.ArgumentParser(description = "Detect faces and five facial landmarks in given image.")
parser.add_argument("--url", type=str, help='Link to image', )
parser.add_argument("--resize_width", type=int, help='Specify width of the image after preprocessing; Default:600 (Recommended)', default=600)
parser.add_argument("--gamma", type=int, help='Gamma Correction Value; Default:1.6 (Recommended)',default=1.6)

args = parser.parse_args()


# Declarations

mtcnn = MTCNN()
path = "/home/kanishk/fd/out/"
url = args.url
img = helper.read_from_url(url)

width = args.resize_width
gamma = args.gamma


# PreProcessing Image

# Gamma Correction
img = preprocess.adjust_gamma(img, gamma)
# CLAHE
img = preprocess.apply_clahe(img)
# Resize Image
img = preprocess.resize_image(img, width)


# Detection

#Detect Faces in the processed image
result = mtcnn.detect_faces(img)
#Convert Bounding Boxes to Square
result = bbops.square_bbox(result)


# Display

#Draw Faces on a copy of the image
img_p = img.copy()
img_p = bbops.draw_bbox(result,img_p)
#cv2.imshow("beep boop", img_p)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# Store
faces = bbops.face_chop(result, img)
for i in range(len(faces)):
	cv2.imwrite( path + str(i+1) + '.jpg', faces[i])
