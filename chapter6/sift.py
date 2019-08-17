import cv2

img = cv2.imread('../images/saxon_eagle.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(51, 163, 236))

cv2.imshow('sift_keypoints', img)
while True:
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

cv2.destroyAllWindows()