import numpy as np
from scipy import ndimage
from utils import imageIO as io
import a7

def test_ComputeTensor():
  im=io.imread('pano/stata-1.png', 1.0)
  tensor=a7.computeTensor(im)
  foo=max(tensor.flatten())
  io.imwrite(tensor/foo, 'tensor_vis.png', 1.0)
  io.imwrite(a7.BW2(im), 'tensor_visBW.png', 1.0)
  
def test_cornerResponse():
  im=io.imread('pano/stata-1.png', 1.0)
  resp=a7.cornerResponse(im)
  
  foo=max(resp.flatten())
  io.imwrite(_magic123(resp/foo),'resp.png', 1.0)

def test_HarrisCorners():
  im=io.imread('pano/stata-1.png', 1.0)
  corners=a7.HarrisCorners(im)
  io.imwrite(_magicCorners(corners, im), 'corners.png', 1.0)
  

def test_computeFeatures():
  im=io.imread('pano/stata-1.png', 1.0)
  corners=a7.HarrisCorners(im)
  features=a7.computeFeatures(im, corners)
  io.imwrite(_magicShowFeatures(features, im*0.5, 4), 'features.png', 1.0)

def test_findCorrespondence():
  im1=io.imread('pano/stata-1.png', 1.0)
  im2=io.imread('pano/stata-2.png', 1.0)
  corners1=a7.HarrisCorners(im1)
  features1=a7.computeFeatures(im1, corners1)
  corners2=a7.HarrisCorners(im2)
  features2=a7.computeFeatures(im2, corners2)
  correspondences = a7.findCorrespondences(features1, features2)
  np.save('corrs', correspondences)
  io.imwrite(_magicDrawCorrespondences(correspondences, im1, im2), \
     'correspondence.png', 1.0)

def test_RANSAC():
  im1=io.imread('pano/stata-1.png', 1.0)
  im2=io.imread('pano/stata-2.png', 1.0)
  correspondences=np.load('corrs.npy')
  H, inliers =a7.RANSAC(correspondences)
  io.imwrite(_magicDrawCorrespondences(correspondences, im1, im2, inliers), \
     'correspondence_ransac.png', 1.0)
def test_autostitch():
  im1=io.imread('pano/stata-1.png', 1.0)
  im2=io.imread('pano/stata-2.png', 1.0)
  im_list=[im1, im2]
  out=a7.autostitch(im_list, 0) 
  io.imwrite(out, 'panorama.png', 1.0)
def test_autostitch2():
  im1=io.imread('pano/guedelon-1.png', 1.0)
  im2=io.imread('pano/guedelon-2.png', 1.0)
  im3=io.imread('pano/guedelon-3.png', 1.0)
  im4=io.imread('pano/guedelon-4.png', 1.0)
  im_list=[im1, im2, im3, im4]
  out=a7.autostitch(im_list, 1) 
  io.imwrite(out, 'panorama2.png', 1.0)

def test_linear_blending():
  im1=io.imread('pano/stata-1.png', 1.0)
  im2=io.imread('pano/stata-2.png', 1.0)
  im_list=[im1, im2]
  out=a7.linear_blending(im_list, 0) 
  io.imwrite(out, 'linear_blending.png', 1.0)

def test_linear_blending2():
  im1=io.imread('pano/guedelon-1.png', 1.0)
  im2=io.imread('pano/guedelon-2.png', 1.0)
  im3=io.imread('pano/guedelon-3.png', 1.0)
  im4=io.imread('pano/guedelon-4.png', 1.0)
  im_list=[im1, im2, im3, im4]
  out=a7.linear_blending(im_list, 1) 
  io.imwrite(out, 'linear_blending2.png', 1.0)

def test_two_scale_blending():
  im1=io.imread('pano/stata-1.png', 1.0)
  im2=io.imread('pano/stata-2.png', 1.0)
  im_list=[im1, im2]
  out=a7.two_scale_blending(im_list, 0) 
  io.imwrite(out, 'two_scale_blending.png', 1.0)

def test_two_scale_blending2():
  im1=io.imread('pano/guedelon-1.png', 1.0)
  im2=io.imread('pano/guedelon-2.png', 1.0)
  im3=io.imread('pano/guedelon-3.png', 1.0)
  im4=io.imread('pano/guedelon-4.png', 1.0)
  im_list=[im1, im2, im3, im4]
  out=a7.two_scale_blending(im_list, 1) 
  io.imwrite(out, 'two_scale_blending2.png', 1.0)



# Helpers for visualization

def _magicDrawCorrespondences(correspondences, im1, im2, correctness=None):
  def max_dim(pt1, pt2):
    return max(abs(pt1.x-(pt2.x+im1.shape[1])), abs(pt1.y-pt2.y))

  def simpleRasterize(c, corre):

    def drawPix(x,y):
      if corre is None:
        im_c[int(y), int(x), 0]=1
        im_c[int(y), int(x), 1]=1
      elif corre: 
        im_c[int(y), int(x), 1]=1
      else:
        im_c[int(y), int(x), 0]=1

    pt2x_ext = c.pt2.x+im1.shape[1]
    
    num_samples=max_dim(c.pt1, c.pt2)+1 
    xx=np.linspace(c.pt1.x, pt2x_ext, num_samples)
    yy=np.linspace(c.pt1.y, c.pt2.y, num_samples)
    map(drawPix, xx, yy)

  if correctness is None:
    correctness=np.repeat(None, len(correspondences))
    
  im_c=np.hstack([im1, im2])
  map(simpleRasterize, correspondences, correctness) 
  return im_c
  

def _magicShowFeatures(features, im_in, rad):
  
  def overlay(f):
    patch=f.descriptor.reshape([2*rad+1, 2*rad+1])
    patch_r=patch*0
    patch_g=patch*0
    patch_r[patch<0]=1
    patch_g[patch>=0]=1
    im[f.pt.y-rad:f.pt.y+rad+1, f.pt.x-rad:f.pt.x+rad+1,0]=patch_r
    im[f.pt.y-rad:f.pt.y+rad+1, f.pt.x-rad:f.pt.x+rad+1,1]=patch_g
    
  im=im_in.copy()
  map(overlay, features)
  return im


def _magicShowPoints(points, im_dim):
  result=np.zeros(im_dim)
  for pt in points:
    result[pt.y, pt.x]=1
  return result

def _magicCorners(corner, im):
  im_corners=_magicShowPoints(corner, im.shape[0:2])
  im_corners=ndimage.filters.\
      convolve(im_corners, np.ones([3,3]), mode='reflect')
  return _magic123(im_corners)+0.5*im

def _magic123(im):
  img=im.copy()
  img = img[:, :, np.newaxis]
  img = np.repeat(img, 3, axis=2)
  return img
  return 

#===Tests=====

test_ComputeTensor()
##test_cornerResponse()
##test_HarrisCorners()
##test_computeFeatures()
##test_findCorrespondence()
##test_RANSAC()
##test_autostitch()
##test_autostitch2()
##test_linear_blending()
##test_linear_blending2()
##test_two_scale_blending()
##test_two_scale_blending2()

  
