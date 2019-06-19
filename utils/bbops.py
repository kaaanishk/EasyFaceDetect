import cv2
import numpy as np

import utils.preprocess as preprocess


def draw_bbox(mtcnn_bbox, input_image):
  """
  This function is used to draw bounding boxes on the
  given image using the results from the MTCNN detector.
  
  Args:
    
    mtcnn_bbox:   JSON formatted result that is obtained after
                  running the MTCNN face detector;
                  This variable contains the labels and coordinates
                  of the bounding boxes.
    input_image:  image on which the bounding boxes are drawn
    
  Returns:
    
    input_image: image with the bounding boxes drawn on
  
  """
  
  for i in range(0,len(mtcnn_bbox)):
    bounding_box = mtcnn_bbox[i]['box']
    keypoints = mtcnn_bbox[i]['keypoints']

    cv2.rectangle(input_image,
                  (bounding_box[0],  bounding_box[1]),
                  (bounding_box[0] + bounding_box[2],
                   bounding_box[1] + bounding_box[3]),
                  (0,155,255), 2)
    cv2.circle(input_image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(input_image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(input_image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(input_image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(input_image,(keypoints['mouth_right']), 2, (0,155,255), 2)
    
  return input_image



def square_bbox(mtcnn_bbox):
  """
  This function is used convert the bounding boxes to square in order to
  maintain a standard size accross all detections.
  
  Args:
    
    results_mtcnn: JSON formatted list of bounding boxes
    
  Returns:
  
    square_bbox: JSON formatted list of square bounding boxes
  
  """
  
  square_bbox = mtcnn_bbox.copy()
  
  for i in range(0, len(mtcnn_bbox)):
    
    bounding_box = mtcnn_bbox[i]['box']
    width = bounding_box[2]
    height = bounding_box[3]
    
    side = np.maximum(width, height)
    
    if side == width:
      # change height
      bounding_box[3] = side
      bounding_box[1] = int( bounding_box[1] + ( height - side ) * 0.5 )
    else:
      # change width
      bounding_box[2] = side
      bounding_box[0] = int( bounding_box[0] + ( width - side ) * 0.5 )
  
  return square_bbox



def face_chop(mtcnn_bbox, image):
  """
  This function returns an array of cropped faces in a given image.
  
  Args:
    
    mtcnn_bbox: JSON formatted list of bounding boxes
    image: input image with faces
    
  Returns:
    
    faces: array of cropped faces
  
  """
  
  faces = np.empty( [len(mtcnn_bbox), 112, 112, 3], dtype = int )
  
  for i in range(0,len(mtcnn_bbox)):
    
    bounding_box = mtcnn_bbox[i]['box']
    
    slice_face = image[ bounding_box[1]:(bounding_box[1]+bounding_box[3]),
                        bounding_box[0]:(bounding_box[0]+bounding_box[2])]
    small_face = preprocess.resize_image_absolute(slice_face, 112, 112)
    
    faces[i][:][:][:] = small_face.copy()
    
  return faces

