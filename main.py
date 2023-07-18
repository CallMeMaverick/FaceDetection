import cv2

# load the classifier
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Create a VideoCapture object
imported_image = cv2.VideoCapture('Elon2.jpeg')

# Read a frame of image
result, image = imported_image.read()
# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# seek for a face
faces = detect.detectMultiScale(gray, 1.3, 5)

# drawing a bounding box
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 3)


cv2.imshow("Elon Image", image)
# define how long your window will last
cv2.waitKey(0)
# release particular image
imported_image.release()
# close all previously opened windows
cv2.destroyAllWindows()
