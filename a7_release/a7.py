import numpy as np
import scipy
import random
import math
import a6
from scipy import ndimage, linalg


class point:
  
    def __init__(self, x, y):
        self.x = x
        self.y = y


class feature:
  
    def __init__(self, pt, descriptor):
        self.pt = pt
        self.descriptor = descriptor


class correspondence:
  
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2


########### HELPER FUNCTIONS ###########

def imIter(im):
  
    for y in xrange(im.shape[0]):
        for x in xrange(im.shape[1]):
            yield (y, x)


def black_image2D(im):
  
    return np.zeros([np.shape(im)[0], np.shape(im)[1]])


def black_image3D(im):
  
    return np.zeros([np.shape(im)[0], np.shape(im)[1], 3])


def BW2D(im, weights=[0.3, 0.6, 0.1]):
  
    out = np.zeros([np.shape(im)[0], np.shape(im)[1]])
    for (y, x) in imIter(out):
        out[y, x] = np.dot(im[y, x], weights)
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
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    lum = BW2D(im)
    blur = ndimage.filters.gaussian_filter(lum, sigmaG)
    horizontal = ndimage.filters.convolve(blur, sobel, mode='reflect')
    vertical = ndimage.filters.convolve(blur, sobel.T, mode='reflect')
    for (y, x) in imIter(out):
        out[y, x] = np.array([horizontal[y, x] ** 2, horizontal[y, x] * vertical[y, x], vertical[y, x] ** 2])
    return ndimage.filters.gaussian_filter(out, [sigmaG * factorSigma, sigmaG * factorSigma, 0])


def cornerResponse(im, k=0.15, sigmaG=1, factorSigma=4):
    '''resp: 2D array charactering the response'''

    out = black_image2D(im)
    tensor = computeTensor(im, sigmaG, factorSigma)
    for (y, x) in imIter(tensor):
        pixel = tensor[y, x]
        M = np.array([[pixel[0], pixel[1]], [pixel[1], pixel[2]]])
        R = np.linalg.det(M) - k * np.trace(M) ** 2
        if R > 0:
            out[y, x] = R
    return out


def HarrisCorners(im, k=0.15, sigmaG=1, factor=4, maxiDiam=7, boundarySize=5):
    '''result: a list of points that locate the images' corners'''

    corner_response = cornerResponse(im, k, sigmaG, factor)
    window_maxes = ndimage.filters.maximum_filter(corner_response, maxiDiam)
    local_maxes = black_image2D(im)
    local_maxes[corner_response == window_maxes] = 1
    local_maxes[corner_response == 0] = 0
    (height, width) = np.shape(local_maxes)
    local_maxes[0:boundarySize] = local_maxes[height - boundarySize:height] = 0
    local_maxes[:, 0:boundarySize] = local_maxes[:, width - boundarySize:width] = 0
    patches = []
    for (y, x) in imIter(local_maxes):
        if local_maxes[y, x] == 1:
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

    return blurredIm[P.y - radiusDescriptor:P.y + radiusDescriptor + 1, P.x - radiusDescriptor:P.x + radiusDescriptor + 1].flatten()


def findCorrespondences(listFeatures1, listFeatures2, threshold=1.7):
    '''correpondences: a list of correspondences object that associate two feature lists.'''

    correspondences = []
    for feature1 in listFeatures1:
        best_correspondence = float('inf')
        best_feature = None
        best_correspondence2 = float('inf')
        best_feature2 = None
        for feature2 in listFeatures2:
            distance = np.sum((feature1.descriptor - feature2.descriptor) ** 2)
            if distance < best_correspondence2:
                best_correspondence2 = distance
                best_feature2 = feature2
                if distance < best_correspondence:
                    best_correspondence2 = best_correspondence
                    best_correspondence = distance
                    best_feature2 = best_feature
                    best_feature = feature2
        if best_correspondence2 / best_correspondence > threshold ** 2:
            correspondences.append(correspondence(feature1.pt, best_feature.pt))
    return correspondences


def RANSAC(listOfCorrespondences, Niter=1000, epsilon=4, acceptableProbFailure=1e-9):
    '''H_best: the best estimation of homorgraphy (3-by-3 matrix)
      inliers: list of booleans that describe whether the element in listOfCorrespondences 
               an inlier or not'''

    number_of_correspondences = len(listOfCorrespondences)
    H_best = None
    inliers = []
    number_of_inliers = 0
    for i in range(Niter):
        feature_pairs = random.sample(listOfCorrespondences, 4)
        homography = a6.computeHomography(A7PairsToA6Pairs(feature_pairs))
        loop_inliers = []
        for correspondence in listOfCorrespondences:
            transformed_point = np.dot(homography, A7PointToA6Point(correspondence.pt1))
            transformed_point[0] /= transformed_point[2]
            transformed_point[1] /= transformed_point[2]
            target_point = A7PointToA6Point(correspondence.pt2)
            loop_inliers.append(math.sqrt(np.sum((transformed_point - target_point) ** 2)) < epsilon)
        loop_inliers_count = loop_inliers.count(True)
        if loop_inliers_count > number_of_inliers:
            number_of_inliers = loop_inliers_count
            inliers = loop_inliers
            H_best = homography
        if ((1 - (loop_inliers_count / number_of_correspondences) ** 4) ** i) < acceptableProbFailure:
            break
    return (H_best, inliers)


def computeNHomographies(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
    '''H_list: a list of Homorgraphy relative to L[refIndex]'''

    number_of_images = len(L)
    homographies = []
    for i in xrange(number_of_images - 1):
        features1 = computeFeatures(L[i], HarrisCorners(L[i]), blurDescriptor, radiusDescriptor)
        features2 = computeFeatures(L[i + 1], HarrisCorners(L[i + 1]), blurDescriptor, radiusDescriptor)
        correspondences = findCorrespondences(features1, features2)
        homographies.append(RANSAC(correspondences)[0])
    compoundHomographies = [np.zeros([3, 3])] * number_of_images
    for i in xrange(number_of_images):
        if i < refIndex:
            compound = np.dot(np.identity(3), homographies[i])
            for j in xrange(i + 1, refIndex):
                compound = np.dot(compound, homographies[j])
            compoundHomographies[i] = compound
        elif i > refIndex:
            compound = np.dot(linalg.inv(np.identity(3)), linalg.inv(homographies[i - 1]))
            for j in xrange(i - 2, refIndex - 1, -1):
                compound = np.dot(compound, linalg.inv(homographies[j]))
            compoundHomographies[i] = compound
        else:
            compoundHomographies[i] = np.identity(3)
    return compoundHomographies


def autostitch(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
    '''Use your a6 code to stitch the images. You need to hand in your A6 code'''

    H_list = computeNHomographies(L, refIndex, blurDescriptor, radiusDescriptor)
    return a6.compositeNImages(L, H_list)


def weight_map(h, w):
    ''' Given the image dimension h and w, return the hxwx3 weight map for linear blending'''

    w_map = np.zeros([h, w, 3])
    h_2 = float(h) / 2
    w_2 = float(w) / 2
    for (y, x) in imIter(w_map):
        w_map[y, x] = (1 - abs(y - h_2) / h_2) * (1 - abs(x - w_2) / w_2)
    return w_map


def linear_blending(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
    ''' Return the stitching result with linear blending'''

    homographies = computeNHomographies(L, refIndex, blurDescriptor, radiusDescriptor)
    (height, width, rgb) = np.shape(L[0])
    weights = weight_map(height, width)
    return compositeNImages_blended(L, homographies, weights)


def two_scale_blending(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
    ''' Return the stitching result with two scale blending'''
    
    homographies = computeNHomographies(L, refIndex, blurDescriptor, radiusDescriptor)
    (height, width, rgb) = np.shape(L[0])
    weights = weight_map(height, width)
    low_frequencies = []
    high_frequencies = []
    for img in L:
        img_low_frequency = ndimage.filters.gaussian_filter(img, [2, 2, 0])     
        low_frequencies.append(img_low_frequency)
        high_frequencies.append(img - img_low_frequency)
    composite_low_frequency = compositeNImages_blended(low_frequencies, homographies, weights)
    composite_high_frequency = compositeNImages_blended(high_frequencies, homographies, weights, True)
    return composite_low_frequency + composite_high_frequency


def compositeNImages_blended(listOfImages, listOfHomographies, weights, onlyHighestWeight = False):
    
    bbox = a6.computeTransformedBBox(np.shape(listOfImages[0]), listOfHomographies[0])
    for (i, H) in enumerate(listOfHomographies):
        bboxI = a6.computeTransformedBBox(np.shape(listOfImages[i]), H)
        bbox = a6.bboxUnion(bboxI, bbox)
    height = bbox[1][0] - bbox[0][0]
    width = bbox[1][1] - bbox[0][1]
    out = np.zeros([height, width, 3])
    out_weight = np.zeros([height, width, 3])
    translation = a6.translate(bbox)
    for i in xrange(len(listOfImages)):
        applyHomographyFast_blended(listOfImages[i], out, np.dot(translation, listOfHomographies[i]), out_weight, weights, onlyHighestWeight)
    out_weight[out_weight == 0] = 1e-12
    return out / out_weight


def applyHomographyFast_blended(source, out, H, out_weight, weight_map, onlyHighestWeight = False):
    '''Takes the image source, warps it by the homography H, and adds it to the composite out.
    This version should only iterate over the pixels inside the bounding box of source's image in out. '''

    bbox = a6.computeTransformedBBox(np.shape(source), H)
    for y in xrange(bbox[0][0], bbox[1][0]):
        for x in xrange(bbox[0][1], bbox[1][1]):
            (yp, xp, wp) = np.dot(linalg.inv(H), np.array([y, x, 1]))
            (ypp, xpp) = (yp / wp, xp / wp)
            if a6.within_bounds(source, ypp, xpp):
                if not onlyHighestWeight:
                    out[y, x] += a6.interpolateLin(source, ypp, xpp) * weight_map[ypp, xpp]
                    out_weight[y, x] += weight_map[ypp, xpp]
                elif np.sum(weight_map[ypp, xpp]) > np.sum(out_weight[y, x]):
                    out[y, x] = a6.interpolateLin(source, ypp, xpp) * weight_map[ypp, xpp]
                    out_weight[y, x] = weight_map[ypp, xpp]
                else:
                    pass


# Helpers, you may use the following scripts for convenience.

def A7PointToA6Point(a7_point):
    return np.array([a7_point.y, a7_point.x, 1.0], dtype=np.float64)


def A7PairsToA6Pairs(a7_pairs):
    A7pointList1 = map(lambda pair: pair.pt1, a7_pairs)
    A6pointList1 = map(A7PointToA6Point, A7pointList1)
    A7pointList2 = map(lambda pair: pair.pt2, a7_pairs)
    A6pointList2 = map(A7PointToA6Point, A7pointList2)
    return zip(A6pointList1, A6pointList2)



