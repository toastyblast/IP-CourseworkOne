# -*- coding: utf-8 -*-
"""
This file includes the code for CourseWork 1 - Task 3.
This involved the drawing of all shapes offered in OpenCV and using these to draw some "art".

@author: Yoran Kerbusch
"""

import numpy as np
import cv2

# Added * 255 because just np.ones() would not create a white background, due to it not recognising the range.
img = np.ones((1024, 1024, 3), np.uint8) * 255

cv2.line(img, (150, 200), (250, 150), (255, 0, 0), 6)
cv2.line(img, (150, 175), (250, 175), (0, 255, 0), 2)
cv2.line(img, (250, 200), (150, 150), (0, 0, 255), 6)
cv2.circle(img, (200, 175), 75, (0, 0, 0), 4)

# Show that the order at which figures are drawn matter for order of appearance.
# The orange circle is drawn afte the gray one, so the orange appears on top of the gray.
# Sidenote: by giving -1 (or any other negative number) as line thickness for any shape, it will fill in the shape.
cv2.circle(img, (700, 600), 50, (127, 127, 127), -1)
cv2.circle(img, (610, 640), 80, (0, 165, 255), -1)

# Draw an ellipse, a.k.a. a deformed circle or you can create other cut circles.
cv2.ellipse(img, (200, 725), (100, 50), 0, 0, 135, (161, 91, 136), -1)
cv2.ellipse(img, (150, 750), (105, 43), 35, 0, 360, (0, 0, 0), 6)
# 270 is the angle the whole ellipse is rotated. The 45 is the start angle and the 135 is the end angle.
cv2.ellipse(img, (100, 675), (40, 40), 270, 45, 135, (85, 140, 61), 3)

# Draw the Dutch flag with an Dutch Royal Orange outline.
cv2.rectangle(img, (300, 100), (750, 200), (40, 28, 174), -1)
cv2.rectangle(img, (300, 200), (750, 300), (255, 255, 255), -1)
cv2.rectangle(img, (300, 300), (750, 400), (139, 70, 33), -1)
cv2.rectangle(img, (295, 95), (755, 405), (0, 79, 255), 5)

# Finally, draw some polygons.
# Polygons take an input array, with the x and y coordinates of each point, connecting these with a line in the order of the array.
# These values must be set to 32 Int values and reshaped.
# Polygons do not have to be closed and can overlap their lines.
pts = np.array([[750, 675], [800, 675], [775, 725], [900, 775], [775, 700]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], False, (0, 0, 0), 3)

# Closed polygon, meaning it connects the first and last point with a line as well, if they don't end on the same coordinates.
pts2 = np.array([[710, 805], [720, 830], [770, 820], [750, 810]], np.int32)
pts2 = pts2.reshape((-1, 1, 2))
cv2.polylines(img, [pts2], True, (127, 127, 127), 3)

# Display name at letter height 24px (scale 1.0) with a complex serif font. Text is drawn with the origin coordinates being on the top-left.
cv2.putText(img, "Yoran Kerbusch (24143341)", (10, 34), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA, False)

# Finally, show the image in the actual window.
cv2.imshow("image", img)

# Show the window until the user presses the "esc" key.
while 1:
    k = cv2.waitKey(0)
    # Only exit when the "esc" key is pressed.
    if k == 27:
        cv2.destroyAllWindows()
        break