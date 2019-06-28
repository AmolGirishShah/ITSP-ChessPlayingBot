from __future__ import division
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
import math
from operator import add

chess_img = cv2.imread('board11.jpeg')  #Test image
chess_img_final = cv2.imread('board11.jpeg')

gray = cv2.cvtColor(chess_img,cv2.COLOR_BGR2GRAY)
corners = []
ret , corners = cv2.findChessboardCorners(gray,(7,7), None)
# 'corners' contains the internal 7x7=49 corners of the chessboard.
# The board has to be empty for the findChessboardCorners function to work otherwise it does not work
# So we will be initializing the board without pieces before the start of the
# game so that we detect the corners
# Since the board nor the camera will move during the entire game so the corners detected during
# initialisation will remain the same at any point in the game.

if ret == False:
	print('Did not find the internal corners of the chessboard')

# cv2.drawChessboardCorners(chess_img,(7,7),corners,ret)
#
# For example, printing 'corners' gives following result
# [[[740.32825 197.17581]]
# [[743.9483  262.66687]]
#  ....
# [[338.4318  623.48486]]]
#
# Note that the (0,0) is the point in the top-left corner of the image
# The x-axis increases horizontally to the right and the y-axis increases as we move down
#
# We cant be sure of how the findChessboardCorners will find the corners in which order so
# we sort the corners
def sortFirst(val): #Sorts by x-coordinate
	return val[0][0]

def sortSecond(val): #Sorts by y-coordinate
	return val[0][1]

corner_sort = sorted(corners, key = sortFirst) #Sorts by x-coordinate
# 'sorted()' has a key parameter to specify a function to be called on each list element prior to making comparisons.
# Since a list element of 'corners' is  [[740.32825 197.17581]] so element[0] = [740.32825 197.17581]
# will give a array containing the x and y coordinates of the corner

final_corner_sort = []

for i in range(7):
	input_row = slice(7*i,(7*i+7)) # Taking one column of 7 corners
	input_row = corner_sort[input_row]
	column_sort = sorted(input_row , key = sortSecond) # Sorting by the y-coor
	final_corner_sort.append(column_sort)
# Now final_corner_sort has corners in the order from top-left most internal corner and
# the column of corners below it and then the next column of corners to the right by one space

# print(final_corner_sort[0][0][0])
# For testing purpose: Saving the internal corners to excel sheet for readability
# ## convert your array into a dataframe
df = pd.DataFrame.from_records(final_corner_sort , index = ['0','1','2','3','4','5','6'])
# ## save to xlsx file
# filepath = 'my_excel_file.xlsx'
# df.to_excel(filepath, index=True)

# We are finding the top-left to bottom-right diagonal diagonal so that we can extrapolate it
# to find the top-left and bottom-right external corner

diag1 = [] #contains diag internal points of first diag
diag1_x = [] #x-coors of above array
diag1_y = [] #y-coors of above array
for i in range(7):
	diag1.append(df.iloc[i][i][0])
	diag1_x.append(df.iloc[i][i][0][0])
	diag1_y.append(df.iloc[i][i][0][1])

# Now we fit a line according to the diagonal points in the internal corners
z = np.polyfit(diag1_x, diag1_y, 1) #returns the coeffients m and c, in that order, in the equation y= mx + c
slope = z[0] #slope = m = tan(theta)
theta = math.atan(slope)

# Finding the top-left point of the chessboard
# We find the distance between the first 2 diagonal internal corners(D1) and the 2nd-3rd diagonalinternal
# corners(D2) and use this distance to find the distance between the external corner on the diagonal
# and the first diagonal internal corner(D0) using geometric mean
D1 = dist.cdist(diag1[0][np.newaxis], diag1[1][np.newaxis], "euclidean")[0]
D2 = dist.cdist(diag1[1][np.newaxis], diag1[2][np.newaxis], "euclidean")[0]
D0 = D1**2 / D2  #Assuming D0,D1,D2 in Geometric progression
tl = [diag1[0][0] - D0 * math.cos(theta), diag1[0][1] - D0 * math.sin(theta) ]

#Finding the bottom-right point of the chessboard
D6 = dist.cdist(diag1[4][np.newaxis], diag1[5][np.newaxis], "euclidean")[0]
D7 = dist.cdist(diag1[5][np.newaxis], diag1[6][np.newaxis], "euclidean")[0]
D8 = D7**2 / D6
br = [diag1[6][0] + D8 * math.cos(theta), diag1[6][1] + D8 * math.sin(theta) ]

# Now using the second diagonal from top-right to bottom-left we find the two
# external corners along this diagonal
diag2 = []
diag2_x = []
diag2_y = []
for i in range(7):
	diag2.append(df.iloc[i][6-i][0])
	diag2_x.append(df.iloc[i][6-i][0][0])
	diag2_y.append(df.iloc[i][6-i][0][1])

z = np.polyfit(diag2_x, diag2_y, 1)
# print(z)
m = z[0]
theta = math.atan(m)
#Testing:
# for i in range(7):
#     chess_img = cv2.circle(chess_img,tuple(diag2[i]), 5, (0,0,255), -1)
# print(diag2[0])

#Finding the bottom-left point of the chessboard
D1 = dist.cdist(diag2[0][np.newaxis], diag2[1][np.newaxis], "euclidean")[0]
D2 = dist.cdist(diag2[1][np.newaxis], diag2[2][np.newaxis], "euclidean")[0]
D0 = D1**2 / D2
bl = [diag2[0][0] - D0 * math.cos(theta), diag2[0][1] - D0 * math.sin(theta) ]

#Finding the top-right point of the chessboard
D6 = dist.cdist(diag2[4][np.newaxis], diag2[5][np.newaxis], "euclidean")[0]
D7 = dist.cdist(diag2[5][np.newaxis], diag2[6][np.newaxis], "euclidean")[0]
D8 = D7**2 / D6
tr = [diag2[6][0] + D8 * math.cos(theta), diag2[6][1] + D8 * math.sin(theta) ]

#Finding the intersection point of the two lines using Cramer's rule
def line(p1, p2): #line defined by two points p1 and p2
	A = (p1[1] - p2[1])
	B = (p2[0] - p1[0])
	C = (p1[0]*p2[1] - p2[0]*p1[1])
	return A, B, -C

# Finds intersection x,y of two lines L1 and L2 defined by the line function
def intersection(L1, L2):
	D  = L1[0] * L2[1] - L1[1] * L2[0]
	Dx = L1[2] * L2[1] - L1[1] * L2[2]
	Dy = L1[0] * L2[2] - L1[2] * L2[0]
	if D != 0:
		x = Dx / D
		y = Dy / D
		return x,y
	else:
		return False

top_edge = []
for i in range(7):
	L1 = line(df.iloc[i][0][0], df.iloc[i][6][0])
	L2 = line(tl, tr)
	top_edge.append(intersection(L1, L2))
	if top_edge[i]:
		pass
	else:
		print ("No single intersection point detected")
bottom_edge = []
for i in range(7):
	L1 = line(df.iloc[i][0][0], df.iloc[i][6][0])
	L2 = line(bl, br)
	bottom_edge.append(intersection(L1, L2))
	if bottom_edge[i]:
		pass
		# print( "Intersection detected:", bottom_edge[i])
	else:
		print ("No single intersection point detected")

right_edge = []
for i in range(7):
	L1 = line(df.iloc[0][i][0], df.iloc[6][i][0])
	L2 = line(tr, br)
	right_edge.append(intersection(L1, L2))
	if right_edge[i]:
		pass
		# print( "Intersection detected:", right_edge[i])
	else:
		print ("No single intersection point detected")

left_edge = []
for i in range(7):
	L1 = line(df.iloc[0][i][0], df.iloc[6][i][0])
	L2 = line(tl, bl)
	left_edge.append(intersection(L1, L2))
	if left_edge[i]:
		pass
		# print( "Intersection detected:", left_edge[i])
	else:
		print ("No single intersection point detected")

#Bringing it all together: Bringing all points in one format in one place in the Corner_matrix[i][j]
w = 9
h = 9
Corner_matrix = [[0 for x in range(w)] for y in range(h)]
Corner_matrix[0][0] = [ tl[0][0] , tl[1][0]]
Corner_matrix[0][8] = [ tr[0][0] , tr[1][0]]
Corner_matrix[8][0] = [ bl[0][0] , bl[1][0]]
Corner_matrix[8][8] = [ br[0][0] , br[1][0]]
for i in range(1,8):
	Corner_matrix[0][i] = [top_edge[i-1][0][0],top_edge[i-1][1][0] ]
	Corner_matrix[i][0] = [left_edge[i-1][0][0],left_edge[i-1][1][0] ]
	Corner_matrix[i][8] = [right_edge[i-1][0][0],right_edge[i-1][1][0] ]
	Corner_matrix[8][i] = [bottom_edge[i-1][0][0],bottom_edge[i-1][1][0]]

for i in range(1,8):
	for j in range(1,8):
		Corner_matrix[i][j] =  [ df.iloc[j-1][i-1][0][0], df.iloc[j-1][i-1][0][1] ]

for i in range(0,9):
	for j in range(0,9):
		chess_img = cv2.circle(chess_img,(int(Corner_matrix[i][j][0]),int(Corner_matrix[i][j][1])), 6, (30*i,30*j,15*i+15*j) , -1)
# print(Corner_matrix)

corners_df = pd.DataFrame.from_records(Corner_matrix,index = ['0','1','2','3','4','5','6','7','8']  )
# ## For testing:save to xlsx file
# filepath = 'my_excel_file1.xlsx'
# corners_df.to_excel(filepath, index = True)

#This function will transform/warp the region of chessboard so that all the squares are of the same size
def four_point_transform(image, pts):#pts are an array of the four corners [tl, tr, br, bl] in order
	(tl, tr, br, bl) = pts
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	maxDim  = max(maxHeight , maxWidth)
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left order

	dst = np.array([
	[0, 0], [maxDim - 1, 0],[maxDim - 1, maxDim - 1],
	[0, maxDim - 1]], dtype = "float32")
	# dst is list of transformed points
	# compute the perspective transform matrix and then apply it
	transformation_matrix = cv2.getPerspectiveTransform(pts, dst)
	warped = cv2.warpPerspective( image, transformation_matrix, (maxDim, maxDim) )

	return warped

corner_points = [ # Taking the extreme four corners of the chessboard
	tuple(Corner_matrix[0][0]), tuple(Corner_matrix[8][0]),
	tuple(Corner_matrix[8][8]), tuple(Corner_matrix[0][8])
	]
pts = np.array(corner_points , dtype = "float32")
warped = four_point_transform(chess_img_final, pts)

# show the original and warped images
cv2.namedWindow('Warped', cv2.WINDOW_NORMAL)
cv2.imshow('Warped', warped)
# Testing:
# dimensions = warped_1.shape
# print('Dimensions:')
# print(dimensions)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', chess_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
