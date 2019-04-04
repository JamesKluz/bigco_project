# import the necessary packages
import argparse
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from skimage.util import img_as_float
from skimage import io
import time

 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
 
# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))

# loop over the number of segments
for numSegments in (200, 300, 400):
	# apply SLIC and extract (approximately) the supplied number
	# of segments
	start_time = time.time()
	image_small = resize(image, (512, 512))
	#segments = slic(image_small, n_segments = numSegments, sigma = 5)
	segments = slic(image_small, n_segments = numSegments)
 
	# # show the output of SLIC
	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(image_small, segments))
	plt.axis("off")
	end_time = time.time()
	elapsed_time = end_time - start_time
	print(elapsed_time)
 
# show the plots
plt.show()