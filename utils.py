import cv2 as cv
import numpy as np

def find_ellipse(points):
	"""
	param points: Points to which an ellipse is fitted.
	return: Best ellipse in the sense of MSE.
	"""
	ellipse = cv.fitEllipse(points)
	return ellipse

landmark_map = {
				"face_contour": np.arange(0, 17),
				"right_eyebrow": np.arange(17, 22),
				"left_eyebrow": np.arange(22, 27),
				"vertical_nose": np.arange(27, 31),
				"horizontal_nose": np.arange(31, 36),
				"left_eye": np.arange(36, 42),
				"right_eye": np.arange(42, 48),
				"outer_lip": np.arange(48, 60),
				"inner_lip": np.arange(60, 68)
				}