import cv2 as cv

img = cv.imread('Resources\Photos\Group2.jpg')

# cv.imshow('Group 2', img)

def rescaleFrame(frame, scale=0.2):

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

frame_resized = rescaleFrame(img)

gray = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray image', gray)

haar_cascade = cv.CascadeClassifier('haar_cascade_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

for (x,y,w,h) in faces_rect:
    cv.rectangle(frame_resized, (x,y), (x+w, y+h), (0,255,0), thickness = 2)

print(f'Number of faces found = {len(faces_rect)}')

cv.imshow('Detected Faces', frame_resized)


cv.waitKey(0)
