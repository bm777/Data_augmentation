# you should install tensorflow_docs before continue to show data augmentation

import urllib

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
AUTOTUNE = tf.data.experimental.AUTOTUNE
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_datasets as tfds
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 5)
import numpy as np
import cv2

url_img = "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg"

class ImageDataGen():
	def __init__(self, path_image=None):
		global url_img 
		self.name = "Data augentation class"

		if path_image == None:
			self.image_path = tf.keras.utils.get_file("data1.jpg", url_img)
		else:
			print("else statement------------------------")
			self.image_path = path_image

		self.imag_string = tf.io.read_file(self.image_path)
		self.image = tf.image.decode_jpeg(self.imag_string, channels=3)
		print("here is url path--------------", self.image_path)

	def visualisze(self, original, augmented):
	    fig = plt.figure()
	    plt.subplot(1,2,1)
	    plt.title('Original image')
	    plt.imshow(original)
	    
	    plt.subplot(1,2,2)
	    plt.title('Augmented image')
	    plt.imshow(augmented)
	    

	def augment_show_all(self):
		
		fig = plt.figure()
		
		#showing hte first image
		plt.subplot(3,3,1)
		plt.title('Original image')
		plt.imshow(self.image)

		#method list
		methods = ["flip", "grayscale", "saturation", "brightness", "rotation", "sobel_edges"]
		
		flipped = tf.image.flip_left_right(self.image)
		plt.subplot(3,3,2)
		plt.title('Flipped image')
		plt.imshow(flipped)

		grayscaled = tf.image.rgb_to_grayscale(self.image)
		plt.subplot(3,3,3)
		plt.title('Grayscaled image')
		plt.imshow(tf.squeeze(grayscaled))  # import to put it, to remove dim otherwise, it should not work. 
											# Error will be: Invalid shape (x, y, 1) for image data

		saturated = tf.image.adjust_saturation(self.image, 3)
		plt.subplot(3,3,4)
		plt.title('Saturated image')
		plt.imshow(saturated)

		brightness = tf.image.adjust_brightness(self.image, 0.35)
		plt.subplot(3,3,5)
		plt.title('Brightnessed image')
		plt.imshow(brightness)

		rotated = tf.image.rot90(self.image)
		plt.subplot(3,3,6)
		plt.title('Rotated image')
		plt.imshow(rotated)

		r_satured = tf.image.random_saturation(self.image, 5, 10)
		plt.subplot(3,3,7)
		plt.title('R satured image')
		plt.imshow(r_satured)

		img = cv2.imread(self.image_path)
		blurred = cv2.blur(img, (3, 3))
		plt.subplot(3,3,8)
		plt.title('Blurred image')
		plt.imshow(blurred)

		plt.show()


	def augment(self, augment_method="brightness"):
		result = None
		
		if augment_method == 'flip':
			flipped = tf.image.flip_left_right(self.image)
			self.visualisze(self.image, flipped)
			result = flipped
		
		elif augment_method == "grayscale":
			grayscaled = tf.image.rgb_to_grayscale(self.image)
			self.visualisze(self.image, tf.squeeze(grayscaled))
			result = grayscaled

		elif augment_method == "saturation":
			saturated = tf.image.adjust_saturation(self.image, 3) #level of saturation the best value is 3
			self.visualisze(self.image, saturated)
			result = saturated

		elif augment_method == "brightness":
			bright = tf.image.adjust_brightness(self.image, 0.35) #the best result is 0.4 on well captured image
			self.visualisze(self.image, bright)
			result = bright

		elif augment_method == "rotation":
			rotated = tf.image.rot90(self.image)
			self.visualisze(self.image, rotated)
			result = rotated

		return result

	def __str__(self):
		return self.name

if __name__ == '__main__':

	# ======================================================================================
	# To augment growing usage of graphic memory by nvidia graphic card.
	gpus = tf.config.experimental.list_physical_devices('GPU')
	print("------------"+str(len(tf.config.experimental.list_physical_devices('CPU'))))
	print("--------------"+str(len(gpus)))

	# ======================================================================================
	if len(gpus) == 0:
		gpus = tf.config.experimental.list_physical_devices('CPU')

	try:
		tf.config.experimental.set_virtual_device_configuration(gpus[0],
		[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
		
		test = ImageDataGen(path_image="assets/elon.jpg") #default parameter for data augmentation
		#test.augment()
		test.augment_show_all()
		print("Test worked ..")
	except Exception as e:
		raise e
	