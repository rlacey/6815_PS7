import numpy as np
import scipy
from scipy import ndimage

class point():
  def __init__(self, x, y):
    self.x=x
    self.y=y

class feature():
  def __init__(self, pt, descriptor):
    self.pt=pt
    self.descriptor=descriptor

class correspondence():
  def __init__(self, pt1, pt2):
    self.pt1=pt1
    self.pt2=pt2

########### HELPER FUNCTIONS ###########
def imIter(im):
 for y in xrange(im.shape[0]):
    for x in xrange(im.shape[1]):
       yield y, x

def black_image(im):
    return np.zeros([np.shape(im)[0], np.shape(im)[1], 3])       
       
def BW(im, weights=[0.3, 0.6, 0.1]):
   out = np.zeros([np.shape(im)[0], np.shape(im)[1]])
   for y,x in imIter(out):
     out[y,x] = np.dot(im[y,x], weights)
   return out

def BW2(im, weights=[0.3, 0.6, 0.1]):
   out = im.copy()
   (height, width, rgb) = np.shape(out)
   for y in xrange(height):
       for x in xrange(width):
           out[y,x] = np.dot(out[y,x], weights)
   return out
  
def lumiChromi(im):
    imL = im.copy()
    imC = im.copy()
    imL = BW(imL)
    imC = im / imL
    return (imL, imC)
########## END HELPER FUNCTIONS #########    

def computeTensor(im, sigmaG=1, factorSigma=4):
  '''im_out: 3-channel-2D array. The three channels are Ixx, Ixy, Iyy'''
  out = black_image(im)
  sobel = np.array([
                     [-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]
                     ])  
  lum = BW(im)
  blur = ndimage.filters.gaussian_filter(lum, sigmaG)
  horizontal = ndimage.filters.convolve(blur, sobel, mode='reflect')
  vertical = ndimage.filters.convolve(blur, sobel.T, mode='reflect')
  for y,x in imIter(out):
    weight = ndimage.filters.gaussian_filter(im[y,x], sigmaG*factorSigma)
    intensities = np.array([horizontal[y,x]**2, horizontal[y,x]*vertical[y,x], vertical[y,x]**2])    
    out[y,x] += weight * intensities
  return out


def cornerResponse(im, k=0.15, sigmaG=1, factorSigma=4):
  '''resp: 2D array charactering the response'''

  return resp

def HarrisCorners(im, k=0.15, sigmaG=1, factor=4, maxiDiam=7, boundarySize=5):
  '''result: a list of points that locate the images' corners'''

  return result

def computeFeatures(im, cornerL, sigmaBlurDescriptor=0.5, radiusDescriptor=4):
  '''f_list: a list of feature objects'''

  return f_list
  
  #features=map(descriptor, im, cornerL)

def descriptor(blurredIm, P, radiusDescriptor=4):
  '''patch: descriptor around 2-D point P, with size (2*radiusDescriptor+1)^2 in 1-D'''
  return patch

def findCorrespondences(listFeatures1, listFeatures2, threshold=1.7):  
  '''correpondences: a list of correspondences object that associate two feature lists.'''
  return correspondences

def RANSAC(listOfCorrespondences, Niter=1000, epsilon=4, acceptableProbFailure=1e-9):
  '''H_best: the best estimation of homorgraphy (3-by-3 matrix)'''
  '''inliers: A list of booleans that describe whether the element in listOfCorrespondences 
  an inlier or not'''
  ''' 6.815 can bypass acceptableProbFailure'''

  return (H_best, inliers)

def computeNHomographies(L, refIndex, blurDescriptior=0.5, radiusDescriptor=4):
  '''H_list: a list of Homorgraphy relative to L[refIndex]'''
  '''Note: len(H_list) is equal to len(L)'''

  return H_list

def autostitch(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
  '''Use your a6 code to stitch the images. You need to hand in your A6 code'''
  return a6.compositeNImages(L, H_list, False)

def weight_map(h,w):
  ''' Given the image dimension h and w, return the hxwx3 weight map for linear blending'''
  return w_map

def linear_blending(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
  ''' Return the stitching result with linear blending'''
  return out

def two_scale_blending(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
  ''' Return the stitching result with two scale blending'''
  return out

# Helpers, you may use the following scripts for convenience.
def A7PointToA6Point(a7_point):
  return np.array([a7_point.y, a7_point.x, 1.0], dtype=np.float64)


def A7PairsToA6Pairs(a7_pairs):
  A7pointList1=map(lambda pair: pair.pt1 ,a7_pairs)
  A6pointList1=map(A7PointToA6Point, A7pointList1)
  A7pointList2=map(lambda pair: pair.pt2 ,a7_pairs)
  A6pointList2=map(A7PointToA6Point, A7pointList2)
  return zip(A6pointList1, A6pointList2)
  


