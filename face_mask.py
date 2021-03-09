from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from imutils.video import VideoStream
import time
from utils import find_ellipse, landmark_map


def face_mask_detector(image_path="images/cara_juan2.jpg"):
    """
    Function that detects landmarks in the faces present in an image.
    Draws polygonal contours to match facial regions of interest.
    Creates facial mask.
    image_path: path of image to analyze.
    """

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    mask = np.ones((image.shape))*255
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for region in landmark_map:
            points = shape[landmark_map[region]].reshape((-1, 1, 2))
            if region in ["face_contour", "right_eyebrow", "left_eyebrow", "horizontal_nose"]:
                cv2.polylines(mask, [points], False, (0, 0, 0))
            else:
                cv2.fillConvexPoly(mask, points, (0, 0, 0))

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", mask)
    cv2.waitKey(0)

def face_mask_detector_live():
    # initialize dlib's face detector (HOG-based) and then load our
    # trained shape predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    frame = vs.read()
    # loop over the frames from the video stream
    while True:
        # grab the frame from the video stream, resize it to have a
        # maximum width of 400 pixels, and convert it to grayscale
        mask = np.ones((frame.shape)) * 255
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        # loop over the face detections
        for rect in rects:

            # use our custom dlib shape predictor to predict the location
            # of our landmark coordinates, then convert the prediction to
            # an easily parsable NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for region in landmark_map:
                points = shape[landmark_map[region]].reshape((-1, 1, 2))
                if region in ["face_contour", "right_eyebrow", "left_eyebrow", "horizontal_nose", "vertical_nose"]:
                    cv2.polylines(mask, [points], False, (0, 0, 0))
                elif region in ["right_eye", "left_eye", "outer_lip"]:
                    cv2.fillConvexPoly(mask, points, (0, 0, 0))
                elif region in ["inner_lip"]:
                    cv2.fillConvexPoly(mask, points, (255, 255, 255))

        cv2.imshow("Frame", mask)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    face_mask_detector_live()