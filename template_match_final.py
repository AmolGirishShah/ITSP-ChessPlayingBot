"""
This file handles the template matching part andthe main function here template_square_match
returns wheter the square is empty or has a red or green pieces
This code assumes that the dataset of images is stored in three folders:"New_Empty","New_Green","New_Red"
"""


from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# original_filepath = "05_new4.png"
#
def template_match_return_code(original_filepath):
	original_image_loaded = cv2.imread(original_filepath)
	code = template_square_match(original_image_loaded , original_filepath)
	return code

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def ssim(imageA, imageB):
	s = measure.compare_ssim(imageA, imageB, multichannel=True)
	return s

def template_square_match(original_image_loaded,original_filepath):
	codes = {"New_Empty":0,"New_Green":1,"New_Red":2}

	min_mse = 100000000000000
	max_ssim = 0.0
	min_mse_img = None
	max_ssim_img = None
	width = 60
	height = 60
	dim = (width, height)

	for directory_name in ["New_Empty","New_Green","New_Red"]:
		directory = os.fsencode(r"C:\Users\AmolShah\Desktop\itsp\ITSP_IP_main\\"+directory_name)
		for file in os.listdir(directory):
			filename = os.fsdecode(file)
			if filename.endswith(".png"):
				imageB = cv2.imread(directory_name+"\\"+filename)
				imageB = cv2.resize(imageB, dim, interpolation = cv2.INTER_AREA)
				img_mse = mse(original_image_loaded,imageB)
				img_ssim = ssim(original_image_loaded,imageB)
				if img_mse < min_mse:
					min_mse = img_mse
					min_mse_img = directory_name+"\\"+filename
					mse_code = codes[directory_name]
				else:
					continue

				if img_ssim > max_ssim:
					max_ssim = img_ssim
					max_ssim_img = directory_name+"\\"+filename
					ssim_code = codes[directory_name]
				else:
				   continue
			else:
				continue
	# print(min_mse_img)
	# print(min_mse)
	# print(max_ssim_img)
	# print(max_ssim)

	'''
	Uncomment the below code if you want to see each individual square being compared to the image with least mse
	with the original image
	'''

	# fig = plt.figure("Images")
	# images_filepath = ("Original", original_filepath), ("SSIM", max_ssim_img), ("MSE", min_mse_img)
	# plt.suptitle("MSE: %.2f , MSE_Code:%s ,SSIM: %.2f , SSIM_Code: %s" % (min_mse,code_decode(mse_code), max_ssim, code_decode(ssim_code) ))
	# for (i, (name, image_filepath)) in enumerate(images_filepath):
	# 	# show the image
	# 	ax = fig.add_subplot(1, 3, i + 1)
	# 	ax.set_title(name)
	# 	print(name)
	# 	print(image_filepath)
	# 	image = cv2.imread(image_filepath,1)
	# 	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	# 	plt.axis("off")
	# # show the figure
	# plt.show()

	if mse_code == ssim_code:
		return mse_code
	else:
		print("Clash between mse and ssim")
		return mse_code

def code_decode(code):
	if code == 0:
		return "Empty"
	elif code == 1:
		return "Green"
	elif code == 2:
		return "Red"
	else:
		return "Wrong code"

def code_to_text(code):
		if code == 0:
			return ""
		elif code == 1:
			return "G"
		elif code == 2:
			return "R"
		else:
			return "-1"
