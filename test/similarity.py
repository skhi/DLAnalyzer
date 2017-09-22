#from skimage.measure import structural_similarity as ssim
from skimage.measure import compare_ssim as ssim
import numpy as np
import cv2
from PIL import Image
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
        print (m ,s) 
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the images
	plt.show()

#image_a = np.array(Image.open('data/patient1/Ant.L.Shoulder/ImageColl_1/images/3_000_F620_20_S141389_2016-04-25_20.45.19.049.tiff'), dtype = np.float32)
#image_b = np.array(Image.open('data/patient1/Ant.L.Shoulder/ImageColl_1/images/4_000_F660_20_S141389_2016-04-25_20.45.19.713.tiff'), dtype = np.float32)


image_list = glob.glob('/disk1/wiscr_data/Patient8/Burn/Study13/Ant.R.U.Trunk/ImageColl_1/Images/7_*')


image_a = np.array(Image.open(image_list[0]), dtype = np.float32)

data = []
for n, img in enumerate(image_list):
   if n == 0:
      continue
#   if n > 10:
#	break
   image_b = np.array(Image.open(img), dtype = np.float32)
#image_b = np.array(Image.open('/disk1/wiscr_data/Patient8/Burn/Study13/Ant.R.U.Trunk/ImageColl_1/Images/7_311_F855_30_S141389_2016-04-25_21.42.13.769.tiff'), dtype = np.float32)
   #compare_images(image_a ,image_b, "test")
   data.append( ssim(image_a, image_b) )
   
plt.hist(data, normed=False, facecolor='green', bins=40)
plt.xlabel('similarity between the first F855 wavelength and other F855')
plt.ylabel('distribution')
plt.xlim(-1, 1)
plt.savefig('fig.pdf')

#with PdfPages('fig.pdf') as pdf:
 #   #As many times as you like, create a figure fig and save it:
  #  fig = plt.figure()
   # pdf.savefig(fig)
