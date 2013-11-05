def time_spent():
  '''N: # of hours you spent on this one'''
  return 18

def collaborators():
  '''Eg. ppl=['batman', 'ninja'] (use their athena username)'''
  return ['cliu2014']

def potential_issues():
  return 'None'

def extra_credit():
#```` Return the function names you implemended````
#```` Eg. return ['full_sift', 'bundle_adjustment']````
  return None


def most_exciting():
  return 'making a panorama of my own images'

def most_difficult():
  return 'Computing the homographies for the autostiching'

def my_panorama():
  input_images=['castle-1.png', 'castle-2.png', 'castle-3.png','castle-4.png']
  output_images= ['castle_linear_blending', 'castle_panorama', 'castle_two_blending']
  return (input_images, output_images)

def my_debug():
  '''return (1) a string explaining how you debug
  (2) the images you used in debugging.

  Eg
  images=['debug1.jpg', 'debug2jpg']
  my_debug='I used blahblahblah...
  '''
  my_debug = 'The stata image and tower were used for debugging as well as my own weight map outputs'.
  images = ['weight_map_long', 'weight_map_square', 'weight_map_wide']
  return (my_debug, images)
