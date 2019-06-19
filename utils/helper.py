import cv2
import numpy as np
from matplotlib import pyplot as plt

import urllib


def read_from_url(url):
  """
  This function is used to read an image from a given URL.
  
  Args:
  
    url: Link to the image
    
  Returns:
  
    image: a copy of the image from the URL
  
  """
  url_response = urllib.request.urlopen(url)
  image_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
  image = cv2.imdecode(image_array, -1)
  
  return image



def plot_histogram(image):
  """
  This function is used to plot the histogram for any given image.
  
  Args:
  
    image: input image for plotting histogram
    
  Returns:
    
    displays a histogram for the given input
  
  """
  hist, bins = np.histogram(image.flatten(),256,[0,256])
  cdf = hist.cumsum()
  cdf_normalized = cdf * hist.max()/ cdf.max()

  plt.plot(cdf_normalized, color = 'b')
  plt.hist(image.flatten(),256,[0,256], color = 'r')
  plt.xlim([0,256])
  plt.legend(('cdf','histogram'), loc = 'upper left')
  plt.show()
  
