import cv2 as cv

algo = 'MOG2'	# you can chose any algo between MOG2 or KNN
if algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
elif algo == 'KNN':
    backSub = cv.createBackgroundSubtractorKNN()

cap = cv.VideoCapture(0)	# capture webcam object

if not cap.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
while True:
    ret, frame = cap.read()		# start getting feed from webcam
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)	# apply above provided Background Subtraction algo to each frame
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
