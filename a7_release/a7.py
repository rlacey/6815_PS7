import numpy as np
import scipy
from scipy import ndimage, linalg

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
       
def black_image2D(im):
    return np.zeros([np.shape(im)[0], np.shape(im)[1]])  

def black_image3D(im):
    return np.zeros([np.shape(im)[0], np.shape(im)[1], 3]) 
       
def BW2D(im, weights=[0.3, 0.6, 0.1]):
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
  out = black_image3D(im)
  sobel = np.array([
                     [-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]
                     ])  
  lum = BW2D(im)
  blur = ndimage.filters.gaussian_filter(lum, sigmaG)
  horizontal = ndimage.filters.convolve(blur, sobel, mode='reflect')
  vertical = ndimage.filters.convolve(blur, sobel.T, mode='reflect')
  for y,x in imIter(out):    
    out[y,x] = np.array([horizontal[y,x]**2, horizontal[y,x]*vertical[y,x], vertical[y,x]**2])
  return ndimage.filters.gaussian_filter(out, [sigmaG*factorSigma, sigmaG*factorSigma, 0])


def cornerResponse(im, k=0.15, sigmaG=1, factorSigma=4):
  '''resp: 2D array charactering the response'''
  out = black_image2D(im)
  tensor = computeTensor(im, sigmaG, factorSigma)
  for y,x in imIter(tensor):
    pixel = tensor[y,x]
    M = np.array([[pixel[0], pixel[1]],
                  [pixel[1], pixel[2]]])
    R = np.linalg.det(M) - k * np.trace(M) ** 2
    if R > 0:
      out[y,x] = R      
  return out


def HarrisCorners(im, k=0.15, sigmaG=1, factor=4, maxiDiam=7, boundarySize=5):
  '''result: a list of points that locate the images' corners'''
  corner_response = cornerResponse(im, k, sigmaG, factor)
  window_maxes = ndimage.filters.maximum_filter(corner_response, maxiDiam)
  local_maxes = black_image2D(im)
  local_maxes[corner_response == window_maxes] = 1
  local_maxes[corner_response == 0] = 0
  (height, width) = np.shape(local_maxes)
  local_maxes[0:boundarySize:] = local_maxes[height-boundarySize:height] = 0
  local_maxes[:, 0:boundarySize:] = local_maxes[:, width-boundarySize:width:] = 0
  patches = []
  for y,x in imIter(local_maxes):
    if local_maxes[y,x] == 1:
      patches.append(point(x, y))  
  return patches

def computeFeatures(im, cornerL, sigmaBlurDescriptor=0.5, radiusDescriptor=4):
  '''f_list: a list of feature objects'''
  lum = BW2D(im)
  blur = ndimage.filters.gaussian_filter(lum, sigmaBlurDescriptor)
  features = []
  for corner in cornerL:
    patch = descriptor(blur, corner, radiusDescriptor)
    features.append(feature(corner, (patch - np.mean(patch)) / np.std(patch)))
  return features


def descriptor(blurredIm, P, radiusDescriptor=4):
  '''patch: descriptor around 2-D point P, with size (2*radiusDescriptor+1)^2 in 1-D'''
  return blurredIm[P.y-radiusDescriptor:P.y+radiusDescriptor+1, P.x-radiusDescriptor:P.x+radiusDescriptor+1].flatten()


def findCorrespondences(listFeatures1, listFeatures2, threshold=1.7):  
  '''correpondences: a list of correspondences object that associate two feature lists.'''
  correspondences = []
  for feature1 in listFeatures1:
    best_correspondence = float("inf")
    for feature2 in listFeatures2:
      distance = np.sum((feature1.descriptor - feature2.descriptor)**2)
      if distance < best_correspondence:
        best_correspondence = distance
        correspondences.append(correspondence(feature1.pt, feature2.pt))
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
  


