import numpy as np
import math
import ipcv

def map_rotation_scale(src, rotation=0, scale=[1, 1]):
	"""
	Title:
		map_rotation_scale
	Description:
		For given rotation and scale factors, returns maps in the
		shape of the new image with the source coordinates for each pixel.
	Attributes:
		src - source image to be transformed
		rotation - integer of angle for image to be rotated clockwise, in degrees
		scale - list, float, or int of scale factor(s)
		      - if list of length 2, for x and y axes, respectively.
		      - if length 1, float, or int, uses provided scale factor for both axes
	Author:
		Molly Hill, mmh5847@rit.edu
	"""
	#error checking
	if type(src) is not np.ndarray:
		msg = "Source image type must be ndarray."
		raise TypeError(msg)
	if type(rotation) is not int and type(rotation) is not float:
		rotation = 0
		print("Error: Specified rotation angle must be int or float. Defaulting to 0.")
	if type(scale) is not list and type(scale) is not int and type(scale) is not float:
		scale=[1,1]
		print("Error: Scale parameter must be list or int. Defaulting to uniform scaling.")
	elif type(scale) is list:
		if len(scale) > 2 or len(scale) == 0:
			scale=[1,1]
			print("Error: Scale parameter list length must be 1 or 2. Defaulting to uniform scaling.")
		elif len(scale) == 1:
			scale += scale
	else:
		scale = [scale,scale]


	#initialize variables
	radians = np.radians(rotation)
	dim = src.shape[0:2]
	
	#create scale, rotation, and transformation matrices
	scaleMax = np.asmatrix([[1/scale[0],0],[0,1/scale[1]]])
	rotMax = np.asmatrix([[math.cos(radians),-math.sin(radians)],[math.sin(radians),math.cos(radians)]])
	transMax = scaleMax*rotMax
	
	#find center of src for offset
	srcXctr = dim[1]/2
	srcYctr = dim[0]/2
	
	
	#take corners of image to find map dimentsions
	xCorners = [0,dim[1]]
	yCorners = [0,dim[0]]
	xChoices = []
	yChoices = []

	#find map dimensions by putting corners through transformation Matrix
	for x in xCorners:
		for y in yCorners:
			coord = np.asmatrix(np.swapaxes([[x,y]],0,1))
			point = transMax.I*coord
			xChoices.append(point[0])
			yChoices.append(point[1])
	xMin = min(xChoices)
	xMax = max(xChoices)
	yMin = min(yChoices)
	yMax = max(yChoices)
	xDim = xMax - xMin
	yDim = yMax - yMin
	Xctr = xDim/2
	Yctr = yDim/2
	newDim = int(xDim), int(yDim)


	#initialize maps
	map1, map2 = np.indices(newDim)

	for x in range(newDim[0]):
		for y in range(newDim[1]):
			quickX = int(map1[x][y]-Xctr) #extract pixel and zero-center
			quickY = int(Yctr-map2[x][y])
			coord = np.swapaxes([[quickX,quickY]],0,1)
			mapCoord = transMax*coord #transform pixel
			map1[x][y] = mapCoord[0]+srcXctr
			map2[x][y] = srcYctr-mapCoord[1]

	map1 = np.float32(np.transpose(map1)) #switch axes and fix data type for correct interpretation by remap()
	map2 = np.float32(np.transpose(map2))
	
	return (map1, map2)


if __name__ == '__main__':

	import cv2
	import ipcv
	import os.path
	import time

	home = os.path.expanduser('~')
	filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
	filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
	#filename = home + os.path.sep + 'src/python/examples/data/lenna_color.tif'
	#filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
	src = cv2.imread(filename)

	startTime = time.clock()
	map1, map2 = ipcv.map_rotation_scale(src, rotation=30, scale=[1.3,0.8])
	elapsedTime = time.clock() - startTime
	print('Elapsed time (map creation) = {0} [s]'.format(elapsedTime))

	startTime = time.clock()
	dst = cv2.remap(src, map1, map2, cv2.INTER_NEAREST)
	#dst = ipcv.remap(src, map1, map2, ipcv.INTER_NEAREST)
	elapsedTime = time.clock() - startTime
	print('Elapsed time (remapping) = {0} [s]'.format(elapsedTime)) 

	srcName = 'Source (' + filename + ')'
	cv2.namedWindow(srcName, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(srcName, src)

	dstName = 'Destination (' + filename + ')'
	cv2.namedWindow(dstName, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(dstName, dst)

	ipcv.flush()
