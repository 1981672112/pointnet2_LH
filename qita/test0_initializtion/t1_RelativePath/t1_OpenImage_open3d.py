import cv2

img_path = r"./img.png"
img = cv2.imread(img_path)

cv2.namedWindow('myPicture', 0)
cv2.resizeWindow('myPicture', 500, 500)
cv2.imshow('myPicture', img)
cv2.waitKey()
cv2.destroyWindow('myPicture')
