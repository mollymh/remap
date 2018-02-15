import numpy as np
import ipcv
import math as m

def remap(src, map1, map2, interpolation=ipcv.INTER_NEAREST, borderMode=ipcv.BORDER_CONSTANT, borderValue=0):
	"""
	Title:
		remap
	Description:
		Given source image and maps of coordinates, transforms source image into map.
	Attributes:
		src (array) - source image to be mapped
		map1 (array or matrix) - source x coordinates to draw DC from
		map2 (array or matrix) - source y coordinates to draw DC from
		interpolation (int) - type of interpolation to use
		borderMode (int) - how to handle area outside of the source image, with the following options:
					- 0 or ipcv.BORDER_CONSTANT, where background is constant color specified
						by borderValue
					- 1 or ipcv.BORDER_REPLICATE, where background draws value from nearest
						image edge
		borderValue (list, int, or float) - list or integer (if grayscale) specifying constant background color
	Author:
		Molly Hill, mmh5847@rit.edu
	"""
	
	#error checking
	if type(src) is not np.ndarray:
		raise TypeError("Source image type must be ndarray.")
	if (type(map1) is not np.ndarray and type(map1) is not np.matrix) \
	or (type(map2) is not np.ndarray and type(map2) is not np.matrix):
		raise TypeError("Provided maps must be of type ndarray.")
	if map1.size != map2.size:
		raise ValueError("Provided maps must be the same size.")
	if type(interpolation) is not int or interpolation<0:
		print("Interpolation type must be positive integer. Defaulting to INTER_NEAREST.")
		interpolation = 0
	elif interpolation != 0:
		print("Only supports INTER_NEAREST for interpolation. Defaulting to that.")
		interpolation = 0
	if type(borderValue) is not list and type(borderValue) is not int and type(borderValue) is not float:
		raise TypeError("Provided borderValue must be positive int, float, or length-3 list of such.")

	#get max DC for normalization and to check borderValue(s)
	bitDepth = int(str(src.dtype)[4:])
	maxCount = int(m.pow(2,bitDepth))
	chan = src.shape[2]

	#checking and fixing borderValue(s)
	if type(borderValue) == int or type(borderValue) == float:
		borderValue = [borderValue]*chan #turning grayscale value into list
	if len(borderValue) != src.shape[2]:
		raise ValueError("BorderValue list must be length matching number of source image channels.")
	for val in borderValue:
		if type(val) is float:
			val = int(val)
		if type(val) is not int or val<0 or val>(maxCount-1):
			raise ValueError("Provided borderValue(s) must be positive float(s) or integer(s) within max digital count.")


	dim = map1.shape
	dstC = np.empty(dim)
	dst = np.repeat(dstC[:,:,np.newaxis],chan,axis=2) #creates array with same number of bands as src

	
	if borderMode == 1: #border replicate
		np.clip(map1,0,src.shape[1]-1,map1)
		np.clip(map2,0,src.shape[0]-1,map2)
		
	if interpolation == 0: #nearest neighbor interpolation
		map1 = np.around(map1).astype(int).flatten()
		map2 = np.around(map2).astype(int).flatten()
	
	for c in range(chan):
		for n in range(map1.size):
			if map1[n]<0 or map2[n]<0 or map1[n]>=src.shape[1] or map2[n]>=src.shape[0]: #checks for edges
				dstC.flat[n] = borderValue[c]/maxCount
			else:
				dstC.flat[n] = src[map2[n]][map1[n]][c]/maxCount
		dst[:,:,c] = dstC #place mapped image in correct channel

	return dst
	
if __name__ == '__main__':

	import cv2
	import ipcv
	import os.path
	import time

	home = os.path.expanduser('~')
	filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
	filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
	#filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
	#filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
	src = cv2.imread(filename)

	map1, map2 = ipcv.map_rotation_scale(src, rotation=30, scale=[1.3, 0.8])

	startTime = time.clock()
	dst = ipcv.remap(src, map1, map2, interpolation=ipcv.INTER_NEAREST, borderMode=ipcv.BORDER_CONSTANT, borderValue=0)
	elapsedTime = time.clock() - startTime
	print('Elapsed time (remap) = {0} [s]'.format(elapsedTime))

	srcName = 'Source (' + filename + ')'
	cv2.namedWindow(srcName, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(srcName, src)

	dstName = 'Destination (' + filename + ')'
	cv2.namedWindow(dstName, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(dstName, dst)

	ipcv.flush()
