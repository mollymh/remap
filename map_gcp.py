import math as m
import numpy as np

def map_gcp(src, map, srcX, srcY, mapX, mapY, order=1):
	"""
	Title:
		map_gcp
	Description:
		Given source and destination ('map') images, and a set of
		ground control points in the two images, will create a map to
		align the source image with the destination.
	Attributes:
		src - source image to be transformed
		map - image for the source image to be aligned with
		srcX - list of x coordinates for ground control points on source image
		srcY - list of y coordinates for ground control points on source image
		mapX - list of x coordinates for ground control points on map image
		mapY - list of y coordinates for ground control points on map image
		order - order of the polynomial to be used
	Author:
		Molly Hill, mmh5847@rit.edu
	"""
	
	#error checking
	if type(src) is not np.ndarray:
		msg = "Source image type must be ndarray."
		raise TypeError(msg)
	if type(map) is not np.ndarray:
		msg = "Map image type must be ndarray."
		raise TypeError(msg)
	if len(srcX)!=len(srcY) or len(srcX)!=len(mapX) or len(srcX)!=len(mapY):
		msg = "Length of GCP coordinates must match for both axes and images."
		raise ValueError(msg)
	if type(order) is not int or order < 1:
		msg = "Specified order must be a positive integer."
		raise TypeError(msg)

	#determine exponents to use for polynomial equation
	expPairs = []
	for x in range(order+1):
		for y in range(order+1):
			if (x+y) <= order:
				expPairs.append([x,y])
				
	#determine X bar matrix
	Xmtx = np.empty([len(srcX),len(expPairs)])
	
	for k in range(len(expPairs)): #implement polynomial to create matrix
		for n in range(len(srcX)):
			Xmtx[n][k] = m.pow(mapX[n],expPairs[k][0])*m.pow(mapY[n],expPairs[k][1])
	
	#get coefficients using least squares
	Xmtx = np.asmatrix(Xmtx)
	srcX = np.asmatrix(srcX)
	srcY = np.asmatrix(srcY)
	a = (Xmtx.T*Xmtx).I * Xmtx.T*srcX.T
	b = (Xmtx.T*Xmtx).I * Xmtx.T*srcY.T

	dim = [map.shape[1],map.shape[0]]
	size = dim[0]*dim[1]
	poly = np.empty([size,len(expPairs)])
	
	#create new matrix to implement polynomial across entire image
	for k in range(len(expPairs)):
		for n in range(size):
			x = n//dim[1]
			y = n-(x*dim[1])
			poly[n][k] = m.pow(x,expPairs[k][0])*m.pow(y,expPairs[k][1])
			
	poly = np.asmatrix(poly)
	
	#factor in coefficients and reshape as an image
	map1 = (a.T * poly.T).reshape(dim)
	map2 = (b.T * poly.T).reshape(dim)
	
	map1 = np.asarray(np.float32(map1.T)) #fix data type and rotation for correct interpretation by remap()
	map2 = np.asarray(np.float32(map2.T))
	
	return (map1, map2)
	
	
if __name__ == '__main__':

	import cv2
	import ipcv
	import os.path
	import time

	home = os.path.expanduser('~')
	imgFilename = home + os.path.sep + \
		'src/python/examples/data/registration/image.tif'
	mapFilename = home + os.path.sep + \
		'src/python/examples/data/registration/map.tif'
	gcpFilename = home + os.path.sep + \
		'src/python/examples/data/registration/gcp.dat'
	src = cv2.imread(imgFilename)
	map = cv2.imread(mapFilename)

	srcX = []
	srcY = []
	mapX = []
	mapY = []
	linesRead = 0
	f = open(gcpFilename, 'r')
	for line in f:
		linesRead += 1
		if linesRead > 2:
			data = line.rstrip().split()
			srcX.append(float(data[0]))
			srcY.append(float(data[1]))
			mapX.append(float(data[2]))
			mapY.append(float(data[3]))
	f.close()

	startTime = time.clock()
	map1, map2 = ipcv.map_gcp(src, map, srcX, srcY, mapX, mapY, order=2)
	elapsedTime = time.clock() - startTime
	print('Elapsed time (map creation) = {0} [s]'.format(elapsedTime)) 

	startTime = time.clock()
	#dst = cv2.remap(src, map1, map2, cv2.INTER_NEAREST)
	dst = ipcv.remap(src, map1, map2, ipcv.INTER_NEAREST)
	elapsedTime = time.clock() - startTime
	print('Elapsed time (remap) = {0} [s]'.format(elapsedTime)) 

	srcName = 'Source (' + imgFilename + ')'
	cv2.namedWindow(srcName, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(srcName, src)

	mapName = 'Map (' + mapFilename + ')'
	cv2.namedWindow(mapName, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(mapName, map)

	dstName = 'Warped (' + mapFilename + ')'
	cv2.namedWindow(dstName, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(dstName, dst)

	ipcv.flush()
