#imutils: convenience functions to make basic image processing operations easier with OpenCV
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

#parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True, help="name of input directory of images")
ap.add_argument("-o", "--output", type=str, required=True, help="output image")
ap.add_argument("-c", "--crop", type=int, default=0, help="whether to crop")
args = vars(ap.parse_args())

print("loading images...")
imagePaths=sorted(list(paths.list_images(args["images"])))
images=[]

for imagePath in imagePaths:
	image=cv2.imread(imagePath)
	images.append(image)

print("stitching images...")
stitcher=cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched)=stitcher.stitch(images)

#status is 0 if stitching is successful
if status==0:
	#cropping
	if args["crop"]!=0:
		print("cropping...")
		#add a border of 10 pixels around the image
		stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

		#convert to grayscale and threshold
		gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
		#resized = cv2.resize(thresh, (0,0), fx=.1, fy=.1)
		cv2.imwrite("thresh.png", thresh)
		#cv2.waitKey(0)

		#find contours
		cnts=cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts=imutils.grab_contours(cnts)
		c=max(cnts, key=cv2.contourArea)

		mask=np.zeros(thresh.shape, dtype="uint8")
		(x, y, w, h)=cv2.boundingRect(c)
		cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

		#crop rectangular region based on thresholding
		minRect=mask.copy()
		sub=mask.copy()

		while cv2.countNonZero(sub)>0:
			minRect=cv2.erode(minRect, None)
			sub=cv2.subtract(minRect, thresh)

		cnts=cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts=imutils.grab_contours(cnts)
		c=max(cnts, key=cv2.contourArea)
		(x, y, w, h)=cv2.boundingRect(c)

		stitched=stitched[y:y+h, x:x+w]

	cv2.imwrite(args["output"], stitched)

	#cv2.imshow("Stitched", stitched)
	#cv2.waitKey(0)

elif status==1:
	print("Not enough images available to create panorama")
elif status==2:
	print("RANSAC homography estimation failed! Not enough distinguishing keypoints found.")
else:
	print("Image stitching failed!!!".format(status))