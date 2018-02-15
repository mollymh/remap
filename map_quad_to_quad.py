import numpy as np

def map_quad_to_quad(img, map, imgX, imgY, mapX, mapY):
	"""
	Title:
		map_quad_to_quad
	Description:
		Given source image and destination ('map') shape, and a set of
		corners for both, will create a map to align the source image
		with the destination.
	Attributes:
		img - source image to be transformed
		map - image for the source image to be aligned with
		imgX - list of x coordinates for corner points on source image
		imgY - list of y coordinates for corner points on source image
		mapX - list of x coordinates for corner points on map
		mapY - list of y coordinates for corner points on map
	Author:
		Molly Hill, mmh5847@rit.edu
	"""
	
	#error-checking
	if type(img) is not np.ndarray:
		raise TypeError("Source image type must be ndarray.")
	if type(map) is not np.ndarray:
		raise TypeError("Map image type must be ndarray.")
	if len(imgX)!=len(imgY) or len(imgX)!=len(mapX) or len(imgX)!=len(mapY):
		raise ValueError("Number of provided corner points must match for both axes and images.")
	if len(imgX) != 4:
		raise ValueError("Must provide exactly 4 corner points.")

	#create matrices to create map-to-image transformation matrix
	firstPmx = np.matrix([[mapX[0],mapY[0],1,0,0,0,-mapX[0]*imgX[0],-mapY[0]*imgX[0]],\
	[mapX[1],mapY[1],1,0,0,0,-mapX[1]*imgX[1],-mapY[1]*imgX[1]],\
	[mapX[2],mapY[2],1,0,0,0,-mapX[2]*imgX[2],-mapY[2]*imgX[2]],\
	[mapX[3],mapY[3],1,0,0,0,-mapX[3]*imgX[2],-mapY[2]*imgX[3]],\
	[0,0,0,mapX[0],mapY[0],1,-mapX[0]*imgY[0],-mapY[0]*imgY[0]],\
	[0,0,0,mapX[1],mapY[1],1,-mapX[1]*imgY[1],-mapY[1]*imgY[1]],\
	[0,0,0,mapX[2],mapY[2],1,-mapX[2]*imgY[2],-mapY[2]*imgY[2]],\
	[0,0,0,mapX[3],mapY[3],1,-mapX[3]*imgY[3],-mapY[3]*imgY[3]]])
	
	secondPmx = np.matrix(np.append(imgX,imgY))
	
	Pmi = np.reshape(np.concatenate((firstPmx.I * secondPmx.T,np.matrix([1]))),(3,3))

	
	#initialize blank maps
	newDim = map.shape[0:2]
	map1, map2 = np.indices(newDim)
	
	
	#implement camera matrix
	origCoord = np.matrix([map1.flatten(),map2.flatten(),np.ones(map1.shape).flatten()])
	newCoord = Pmi * origCoord
	map1 = np.reshape(newCoord[0,:]/newCoord[2,:],newDim) #divide by homogenous coordinates and reshape
	map2 = np.reshape(newCoord[1,:]/newCoord[2,:],newDim)
			
			
	#switch axes and fix data type for correct interpretation by remap()		
	map1 = np.float32(np.transpose(map1))
	map2 = np.float32(np.transpose(map2))

	return map1, map2

if __name__ == '__main__':

	import cv2
	import ipcv
	import os.path
	import time

	home = os.path.expanduser('~')
	imgFilename = home + os.path.sep + 'src/python/examples/data/lenna.tif'
	mapFilename = home + os.path.sep + 'src/python/examples/data/gecko.jpg'
	img = cv2.imread(imgFilename)
	map = cv2.imread(mapFilename)

	mapName = 'Select corners for the target area (CW)'
	cv2.namedWindow(mapName, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(mapName, map)

	print('')
	print('--------------------------------------------------------------')
	print('  Select the corners for the target area of the source image')
	print('  in clockwise order beginning in the upper left hand corner')
	print('--------------------------------------------------------------')
	p = ipcv.PointsSelected(mapName, verbose=True)
	while p.number() < 4:
		cv2.waitKey(100)
	cv2.destroyWindow(mapName)

	imgX = [0, img.shape[1]-1, img.shape[1]-1, 0]
	imgY = [0, 0, img.shape[0]-1, img.shape[0]-1]
	mapX = p.x()
	mapY = p.y()

	print('')
	print('Image coordinates ...')
	print('   x -> {0}'.format(imgX))
	print('   y -> {0}'.format(imgY))
	print('Target (map) coordinates ...')
	print('   u -> {0}'.format(mapX))
	print('   v -> {0}'.format(mapY))
	print('')

	startTime = time.clock()
	map1, map2 = ipcv.map_quad_to_quad(img, map, imgX, imgY, mapX, mapY)
	elapsedTime = time.clock() - startTime
	print('Elapsed time (map creation) = {0} [s]'.format(elapsedTime)) 

	startTime = time.clock()
	dst = cv2.remap(img, map1, map2, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)
	elapsedTime = time.clock() - startTime
	print('Elapsed time (remap) = {0} [s]'.format(elapsedTime)) 
	print('')

	compositedImage = map
	mask = np.where(dst != 0) #I changed numpy to np just so I could have shortened syntax above
	if len(mask) > 0:
		compositedImage[mask] = dst[mask]

	compositedName = 'Composited Image'
	cv2.namedWindow(compositedName, cv2.WINDOW_AUTOSIZE)
	#cv2.imshow(compositedName, dst)
	cv2.imshow(compositedName, compositedImage)

	ipcv.flush()
