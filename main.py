# Standard imports
import cv2
import numpy as np 

# Read images
src = cv2.imread("Images/plane.jpg")
dst = cv2.imread("Images/sky.jpg")

# Create a rough mask around the airplane.
#src_mask = np.zeros(src.shape, src.dtype)
#src_mask = 255 * np.ones(src.shape, src.dtype)
src_mask = np.zeros(src.shape, src.dtype)

# Of course, we are too lazy to speak, we do not need the following two lines, but the effect is a little worse.
# , then we have to change the above line to mask = 255 * np.ones(obj.shape, obj.dtype) <-- all white
poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
cv2.fillPoly(src_mask, [poly], (255, 255, 255))

# This is where the aircraft CENTER is located
center = (400,400)

# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

# Save results
cv2.imshow("Output", output)
#cv2.imshow("plane", src)
#cv2.imshow("sky", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
