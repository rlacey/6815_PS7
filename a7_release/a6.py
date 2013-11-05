#assignment 6 starter code
#by Abe Davis
#
# Student Name: Ryan Lacey
# MIT Email: rlacey@mit.edu

import math
import numpy as np
from scipy import linalg


############## HELPER FUNCTIONS ###################
def imIter(im):
 for y in xrange(im.shape[0]):
    for x in xrange(im.shape[1]):
       yield y, x

def within_bounds(im, y, x):
    return y>0 and y<im.shape[0] and x>0 and x<im.shape[1]

def clipX(im, x):
   return min(np.shape(im)[1]-1, max(x, 0))

def clipY(im, y):
   return min(np.shape(im)[0]-1, max(y, 0))
              
def getSafePix(im, y, x):
 return im[clipY(im, y), clipX(im, x)]

def interpolateLin(im, y, x, repeatEdge=0):
    # R and P notation from http://en.wikipedia.org/wiki/Bilinear_interpolation
    # Get nearby neighbor coordinates
    leftX = int(math.floor(x))    
    rightX = int(math.ceil(x))
    bottomY = int(math.floor(y))
    topY = int(math.ceil(y))
    # Interpolate x's
    if leftX == rightX:
        R2 = getSafePix(im, topY, x)
        R1 = getSafePix(im, bottomY, x)
    else:
        R2 = getSafePix(im, topY, rightX) * (x - leftX) + getSafePix(im, topY, leftX) * (rightX - x)
        R1 = getSafePix(im, bottomY, rightX) * (x - leftX) + getSafePix(im, bottomY, leftX) * (rightX - x)
    # interpolate midpoints (R1 and R2)
    if bottomY == topY:
        P = R2
    else:
        P = R1 * (topY - y) + R2 * (y - bottomY)
    return P
################# END HELPER ######################

def applyHomography(source, out, H, bilinear=False):
    '''takes the image source, warps it by the homography H, and adds it to the composite out.
    If bilinear=True use bilinear interpolation, otherwise use NN. Keep in mind that we are
    iterating through the output image, and the transformation from output pixels to source
    pixels is the inverse of the one from source pixels to the output. Does not return anything.'''
    for y,x in imIter(out):        
        (yp, xp, wp) = np.dot(linalg.inv(H), np.array([y,x,1]))
        (ypp, xpp) = (yp/wp, xp/wp)
        if within_bounds(source, ypp, xpp):        
            if bilinear:
                out[y,x] = interpolateLin(source, ypp, xpp)
            else:
                out[y,x] = source[ypp,xpp]

def addConstraint(ststm, i, constr):
    '''Adds the constraint constr to the system of equations ststm. constr is simply listOfPairs[i] from the
    argument to computeHomography. This function should fill in 2 rows of systm. We want the solution to our
    system to give us the elements of a homography that maps constr[0] to constr[1]. Does not return anything'''
    (y, x) = (constr[0][0], constr[0][1])
    (yp, xp) = (constr[1][0], constr[1][1])
    ststm[2*i] = np.array([y, x, 1, 0, 0, 0, -1*y*yp, -1*x*yp, -1*yp])
    ststm[2*i+1] = np.array([0, 0, 0, y, x, 1, -1*y*xp, -1*x*xp, -1*xp])

def computeHomography(listOfPairs):
    '''Computes and returns the homography that warps points listOfPairs[-][0] to listOfPairs[-][1]'''
    A = np.zeros([9, 9])
    for i, pair in enumerate(listOfPairs):
        addConstraint(A, i, pair)
    A[8] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    B = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    x = np.dot(linalg.inv(A), B)
    return np.reshape(x, (3,3))

def computeTransformedBBox(imShape, H):
    '''computes and returns [[ymin, xmin],[ymax,xmax]] for the transformed version of the rectangle
    described in imShape. Keep in mind that when you usually compute H you want the homography that
    maps output pixels into source pixels, whereas here we want to transform the corners of our source
    image into our output coordinate system.'''
    # Image corners with weight 1: t=top, b=bottom, l=left, r=right
    tl = np.array([imShape[0], 0, 1.0])
    tr = np.array([imShape[0], imShape[1], 1.0])
    bl = np.array([0, 0, 1.0])
    br = np.array([0, imShape[1], 1.0])
    # Homography of origin (bl) defines base values
    blH = np.dot(H, bl)
    ymin = ymax = blH[0] / blH[2]
    xmin = xmax = blH[1] / blH[2]
    # Shift min and max depending on other corners
    corners = [tl, tr, br]
    for corner in corners:
        cornerH = np.dot(H, corner)
        (yp, xp) = (cornerH[0]/cornerH[2], cornerH[1]/cornerH[2])
        ymin = min(ymin, yp)
        ymax = max(ymax, yp)
        xmin = min(xmin, xp)
        xmax = max(xmax, xp)        
    return [[int(math.floor(ymin)), int(math.floor(xmin))],[int(math.ceil(ymax)),int(math.ceil(xmax))]]

def bboxUnion(B1, B2):
    '''No, this is not a professional union for beat boxers. Though that would be awesome. Rather,
    you should take two bounding boxes of the form [[ymin, xmin,],[ymax, xmax]] and compute their union.
    Return a new bounding box of the same form. Beat boxing optional...'''
    return [[min(B1[0][0],B2[0][0]), min(B1[0][1],B2[0][1])], [max(B1[1][0],B2[1][0]), max(B1[1][1],B2[1][1])]]


def translate(bbox):
    '''Takes a bounding box, returns a translation matrix that translates the top left corner of that
    bounding box to the origin. This is a very short function.'''
    return np.array([np.array([1.0, 0.0, -bbox[0][0]]),
                     np.array([0.0, 1.0, -bbox[0][1]]),
                     np.array([0.0, 0.0, 1.0])])

def stitch(im1, im2, listOfPairs):
    '''Stitch im1 and im2 into a panorama. The resulting panorama should be in the coordinate system of im2,
    though possibly extended to a larger image. That is, im2 should never appear distorted in the resulting
    panorama, only possibly translated. Returns the stitched output (which may be larger than either input image).'''
    H = computeHomography(listOfPairs)
    bbox1 = computeTransformedBBox(np.shape(im1), H)
    bbox = bboxUnion(bbox1, [[0,0], [np.shape(im2)[0], np.shape(im2)[1]]])    
    height = bbox[1][0] - bbox[0][0]
    width = bbox[1][1] - bbox[0][1]
    out = np.zeros([height, width, 3])
    translation = translate(bbox)
    for y,x in imIter(out):
        (yt, xt, wt) = np.dot(linalg.inv(translation), np.array([y,x,1.0]))
        (yp, xp, wp) = np.dot(linalg.inv(H), np.array([yt,xt,wt]))
        (ypp, xpp) = (yp/wp, xp/wp)
        if within_bounds(im1, ypp, xpp):
            out[y,x] = interpolateLin(im1, ypp, xpp)
        elif within_bounds(im2, yt, xt):
            out[y,x] = interpolateLin(im2, yt, xt)
    return out

#######6.865 Only###############

def applyHomographyFast(source, out, H, bilinear=False):
    '''Takes the image source, warps it by the homography H, and adds it to the composite out.
    This version should only iterate over the pixels inside the bounding box of source's image in out. '''
    bbox = computeTransformedBBox(np.shape(source), H)
    for y in xrange(bbox[0][0], bbox[1][0]):
        for x in xrange(bbox[0][1], bbox[1][1]):
            (yp, xp, wp) = np.dot(linalg.inv(H), np.array([y,x,1]))
            (ypp, xpp) = (yp/wp, xp/wp)
            if within_bounds(source, ypp, xpp):        
                if bilinear:
                    out[y,x] = interpolateLin(source, ypp, xpp)
                else:
                    out[y,x] = source[ypp,xpp]


def computeNHomographies(listOfListOfPairs, refIndex):
    homographies = [computeHomography(pair) for pair in listOfListOfPairs]
    compoundHomographies =  [[np.zeros([3,3])] for i in range(len(listOfListOfPairs)+1)]   
    for i in xrange(len(listOfListOfPairs)+1):
        if i < refIndex:
            compound = np.dot(np.identity(3), homographies[i])
            for j in xrange(i+1, refIndex):
                compound = np.dot(compound, homographies[j])
            compoundHomographies[i] = compound
        elif i > refIndex:
            compound = np.dot(linalg.inv(np.identity(3)), linalg.inv(homographies[i-1]))
            for j in xrange(i-2, refIndex, -1):
                compound = np.dot(compound, linalg.inv(homographies[j]))
            compoundHomographies[i] = compound
        else:
            compoundHomographies[i] = np.identity(3)    
    return compoundHomographies 


def compositeNImages(listOfImages, listOfH):
    '''Computes the composite image. listOfH is of the form returned by computeNHomographies.
    Hint: You will need to deal with bounding boxes and translations again in this function.'''
    bbox = computeTransformedBBox(np.shape(listOfImages[0]), listOfH[0])
    # Bounding box to encompass all images in panorama
    for i, H in enumerate(listOfH):
        bboxI = computeTransformedBBox(np.shape(listOfImages[i]), H)
        bbox = bboxUnion(bboxI, bbox)
    height = bbox[1][0] - bbox[0][0]
    width = bbox[1][1] - bbox[0][1]
    out = np.zeros([height, width, 3])    
    translation = translate(bbox)
    for i in xrange(len(listOfImages)):
        applyHomographyFast(listOfImages[i], out, np.dot(translation, listOfH[i]), True)
    return out


def stitchN(listOfImages, listOfListOfPairs, refIndex):
    '''Takes a list of N images, a list of N-1 listOfPairs, and the index of a reference image.
    The listOfListOfPairs contains correspondences between each image Ii and image I(i+1).
    The function should return a completed panorama'''
    listOfH = computeNHomographies(listOfListOfPairs, refIndex)
    return compositeNImages(listOfImages, listOfH)
