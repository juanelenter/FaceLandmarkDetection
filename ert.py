# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from imutils.video import VideoStream
import time

def detector_image(image_path = "images/dinho.jpg"):

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(image_path)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", image)
	cv2.waitKey(0)

def detector_live():
	# initialize dlib's face detector (HOG-based) and then load our
	# trained shape predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	# initialize the video stream and allow the cammera sensor to warmup
	print("[INFO] camera sensor warming up...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	# loop over the frames from the video stream
	while True:
		# grab the frame from the video stream, resize it to have a
		# maximum width of 400 pixels, and convert it to grayscale
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detect faces in the grayscale frame
		rects = detector(gray, 0)
		# loop over the face detections
		for rect in rects:
			# convert the dlib rectangle into an OpenCV bounding box and
			# draw a bounding box surrounding the face
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			# cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			# use our custom dlib shape predictor to predict the location
			# of our landmark coordinates, then convert the prediction to
			# an easily parsable NumPy array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			# loop over the (x, y)-coordinates from our dlib shape
			# predictor model draw them on the image
			for (sX, sY) in shape:
				cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)
				# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()


if __name__ == '__main__':
	detector_live()