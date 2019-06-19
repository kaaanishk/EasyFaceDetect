import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from mtcnn.mtcnn import MTCNN


def adjust_gamma(image, gamma=1.0):
  """
  This function is used to make a Lookup Table (LUT)
  to adjust the gamma value for each pixel accordingly
  
  Args:
    
    image: input image to operate on
    gamma: gamma value;
           gamma < 1 makes the image darker and
           gamma > 1 makes the image brighter;
           default: gamma = 1 which returns unaltered image
  Returns:
  
    image: input image with gamma value changed according to the input
	
  """
  invGamma = 1.0 / gamma
  table = np.array( [((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)] ).astype("uint8")
  
	# Return image which is gamma corrected using the LUT
  return cv2.LUT(image, table)



def apply_clahe(image):
  
  """
  This function is used to convert the input `image` with RGB profile,
  into LAB color profile.
  We do this since Contrast Limited Adaptive Histogram Equalisation(CLAHE)
  is done on grayscale images only, and the 'L' component of the LAB
  color profile helps us isolate the greyscale data while preserving color
  
  Args:
    
    image: input image to operate on
    
  Returns:
    
    img_clahe: image with CLAHE applied
  
  """
  lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  lab_planes = cv2.split(lab)
  clahe = cv2.createCLAHE()
  lab_planes[0] = clahe.apply(lab_planes[0])
  lab = cv2.merge(lab_planes)
  img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
  
  return img_clahe



def tan_triggs(image, alpha=0.1,gamma=0.2, sigma0=1, sigma1=2, tau=10):
  """
  This function is used to apply Tan and Triggs Illumination Normalisation
  on the given image as described in:
  
  X. Tan and B. Triggs, "Enhanced Local Texture Feature Sets for
  Face Recognition Under Difficult Lighting Conditions," in
  IEEE Transactions on Image Processing, vol. 19, no. 6, pp. 1635-1650
  June 2010.
  
  Default parameters are taken from the paper.

  """
  
  image = np.array(image, dtype = np.float32)
  image = np.power(image, gamma)
  
  s0 = 3 * sigma0
  s1 = 3 * sigma1
  
  if (s0 % 2) == 0:
    s0 += 1
  if (s1 % 2) == 0:
    s1 += 1
    
  # Taking Difference of Gaussian Filters (DoG)
  image = np.asarray(ndimage.gaussian_filter( image, sigma0 ) - ndimage.gaussian_filter( image, sigma1))
  
  image = image / np.power( np.mean( np.power( np.abs(image), alpha ) ), 1.0/alpha ) 
  image = image / np.power( np.mean( np.power( np.minimum( np.abs(image), tau ), alpha ) ), 1.0/alpha )
  image = np.tanh( image / tau ) * tau
  image = cv2.normalize( image, image, -220, 0, cv2.NORM_MINMAX )
  
  return np.array( image, np.uint8 )



def resize_image(image, width=600):
  """
  This function is used to resize images given the width in pixels.
  The height of the resized image is calculated based on the given width.
  By default, the width of the output image is set to 600px since
  our model seems to perform best with images of this size.
  
  Args:
    
    image: input image to operate on
    width: desired width of the output image;
           default: 600px;
           This value gives best performance on the
           private dataset used therefore it is set as default.
    
  Returns:
    
    resized_image: image resized based on the desired width.
  
  """
  r = width / image.shape[1]
  dim = (width, int(image.shape[0] * r))
  resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  return resized_image



def resize_image_absolute(image, width, height): 
	dim = (width, height)
	resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized_image

