from __future__ import division
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
import math
from operator import add

chess_img = cv2.imread('board6.png')
kernel = np.ones((5,5), np.uint8)
gray = cv2.cvtColor(chess_img,cv2.COLOR_BGR2GRAY)
corners = []
ret , corners = cv2.findChessboardCorners(gray,(7,7), None)
# if ret==True:
#     print(corners)
if ret == False:
    print('Did not find')
cv2.drawChessboardCorners(chess_img,(7,7),corners,ret)

def sortFirst(val):
    return val[0][0]

def sortSecond(val):
    return val[0][1]

cornersort = corners
#print('Corners:')
#print(cornersort)
cornersort = sorted(cornersort, key = sortFirst)
#print(cornersort)
# print(len(cornersort))
finalsort = []

for i in range(7):
    s = slice(7*i,(7*i+7))
    inparr = cornersort[s] #inputarray
    rowst = sorted(inparr , key = sortSecond) #rowsort
    finalsort.append(rowst)

print('final sort is:')
print(finalsort)
#print(finalsort[0][0][0])

## convert your array into a dataframe
df = pd.DataFrame.from_records(finalsort , index = ['0','1','2','3','4','5','6'])

## save to xlsx file

filepath = 'my_excel_file.xlsx'

df.to_excel(filepath, index=True)
diag1 = []
diag1_x = []
diag1_y = []
for i in range(7):
    diag1.append(df.iloc[i][i][0])
    diag1_x.append(df.iloc[i][i][0][0])
    diag1_y.append(df.iloc[i][i][0][1])

z = np.polyfit(diag1_x, diag1_y, 1)
print(z)
m = z[0]
theta = math.atan(m)
for i in range(7):
    chess_img = cv2.circle(chess_img,tuple(diag1[i]), 5, (0,0,255), -1)
#print(diag1[0])

#Finding the top-left point of the chessboard
D1 = dist.cdist(diag1[0][np.newaxis], diag1[1][np.newaxis], "euclidean")[0]
D2 = dist.cdist(diag1[1][np.newaxis], diag1[2][np.newaxis], "euclidean")[0]
D0 = D1**2 / D2
tl = [diag1[0][0] - D0 * math.cos(theta), diag1[0][1] - D0 * math.sin(theta) ]
print('top left:')
print(tl)
chess_img = cv2.circle(chess_img,tuple(tl), 2, (0,0,255), -1)

#Finding the bottom-right point of the chessboard
D6 = dist.cdist(diag1[4][np.newaxis], diag1[5][np.newaxis], "euclidean")[0]
D7 = dist.cdist(diag1[5][np.newaxis], diag1[6][np.newaxis], "euclidean")[0]
D8 = D7**2 / D6
br = [diag1[6][0] + D8 * math.cos(theta), diag1[6][1] + D8 * math.sin(theta) ]
print('bottom right:')
print(br)
chess_img = cv2.circle(chess_img,tuple(br), 2, (0,0,255), -1)

#
#
#
diag2 = []
diag2_x = []
diag2_y = []
for i in range(7):
    diag2.append(df.iloc[i][6-i][0])
    diag2_x.append(df.iloc[i][6-i][0][0])
    diag2_y.append(df.iloc[i][6-i][0][1])

print(diag2)
z = np.polyfit(diag2_x, diag2_y, 1)
print(z)
m = z[0]
theta = math.atan(m)
for i in range(7):
    chess_img = cv2.circle(chess_img,tuple(diag2[i]), 5, (0,0,255), -1)
print(diag2[0])

#Finding the bottom-left point of the chessboard
D1 = dist.cdist(diag2[0][np.newaxis], diag2[1][np.newaxis], "euclidean")[0]
D2 = dist.cdist(diag2[1][np.newaxis], diag2[2][np.newaxis], "euclidean")[0]
D0 = D1**2 / D2
bl = [diag2[0][0] - D0 * math.cos(theta), diag2[0][1] - D0 * math.sin(theta) ]
print('bottom left:')
print(bl)
chess_img = cv2.circle(chess_img,tuple(bl), 2, (0,255,255), -1)#yellow

#Finding the top-right point of the chessboard
D6 = dist.cdist(diag2[4][np.newaxis], diag2[5][np.newaxis], "euclidean")[0]
D7 = dist.cdist(diag2[5][np.newaxis], diag2[6][np.newaxis], "euclidean")[0]
D8 = D7**2 / D6
tr = [diag2[6][0] + D8 * math.cos(theta), diag2[6][1] + D8 * math.sin(theta) ]
print('top right:')
print(tr)
chess_img = cv2.circle(chess_img,tuple(tr), 2, (255,0,255), -1)#pink

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

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
        print( "Intersection detected:", top_edge[i])
        chess_img = cv2.circle(chess_img,tuple(top_edge[i]), 2, (0,0,255), -1)
    else:
        print ("No single intersection point detected")
bottom_edge = []
for i in range(7):
    L1 = line(df.iloc[i][0][0], df.iloc[i][6][0])
    L2 = line(bl, br)
    bottom_edge.append(intersection(L1, L2))
    if bottom_edge[i]:
        print( "Intersection detected:", bottom_edge[i])
        chess_img = cv2.circle(chess_img,tuple(bottom_edge[i]), 2, (255,150,255), -1)
    else:
        print ("No single intersection point detected")

right_edge = []
for i in range(7):
    L1 = line(df.iloc[0][i][0], df.iloc[6][i][0])
    L2 = line(tr, br)
    right_edge.append(intersection(L1, L2))
    if right_edge[i]:
        print( "Intersection detected:", right_edge[i])
        chess_img = cv2.circle(chess_img,tuple(right_edge[i]), 2, (0,255,255), -1)
    else:
        print ("No single intersection point detected")

left_edge = []
for i in range(7):
    L1 = line(df.iloc[0][i][0], df.iloc[6][i][0])
    L2 = line(tl, bl)
    left_edge.append(intersection(L1, L2))
    if left_edge[i]:
        print( "Intersection detected:", left_edge[i])
        chess_img = cv2.circle(chess_img,tuple(left_edge[i]), 2, (150,150,255), -1)
    else:
        print ("No single intersection point detected")

print(df[0][0][0])
w = 9
h = 9
Matrix = [[0 for x in range(w)] for y in range(h)]
Matrix[0][0] = tl
Matrix[0][8] = tr
Matrix[8][0] = bl
Matrix[8][8] = br
for i in range(1,8):
    Matrix[0][i] = list(top_edge[i-1])
    Matrix[i][0] = list(left_edge[i-1])
    Matrix[i][8] = list(right_edge[i-1])
    Matrix[8][i] = list(bottom_edge[i-1])

print(Matrix)
corners_df = pd.DataFrame.from_records(Matrix,index = ['0','1','2','3','4','5','6','7','8']  )

## save to xlsx file

filepath = 'my_excel_file1.xlsx'
new_array = np.array([1.5, 4.65, 7.845])
print(new_array)

corners_df.to_excel(filepath, index = True)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',chess_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
