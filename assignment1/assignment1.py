import cv2 as cv

#KNN or MOG2 for bg sub:
backSub = cv.createBackgroundSubtractorMOG2( #adding thresholds for MOG2
    history=500, #history is how long background lasts on screen
    varThreshold=50, #varthreshold is sensitivity to change e.g. lower = more motion detected
    detectShadows=True #detectShadows as grey regions or not - removed for binary image but i like the effect more
)

#opening webcam
capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera")
    exit()
    

while True:
    ret, frame = capture.read()
    if not ret or frame is None:
        break
    frame = cv.flip(frame, 1)  
    #mirror for normal webcam view

    #applying bg sub to fg mask
    fgMask = backSub.apply(frame)


    #adding CANNY EDGE DETECTION to fgMask:
    #binary mask from bg sub - so dont need to make greyscale first
    #add blur to clean up noise
    #all attached to fgMask so edges only on moving parts
    #tried Gaussian and median blur - median rid of black and pepper noise better
    
    #cleaning the mask --
    #fgMask = cv.GaussianBlur(fgMask, (3,3), 0) 
    fgMask = cv.medianBlur(fgMask, 5)
    #fgMask = cv.threshold(fgMask, 127, 255, cv.THRESH_BINARY)[1] #makes sure mask is binary

    #edges only on moving parts
    edges = cv.Canny(fgMask, threshold1=10, threshold2=100)


    #adding CONTOURS:
    #makes continuous lines around edges detected
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #edges: output of canny, retrexternal: only external contours, chain: store only endpoints
    
    #adding BOUNDING BOXES to contours:
    #only need if i want to analyse moving edges after -- like object tracking etc.
    for contour in contours:
        if cv.contourArea(contour) > 1000:  #adjust threshold for detecting smaller/larger objects
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x+w, y+h), (250, 229, 202), 2)


    # Define a line (e.g., horizontal line at y=300)
    # line_y = 300
    # cv.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)  # Draw the line

    # for contour in contours:
    #     if cv.contourArea(contour) > 1000:  # Filter small contours
    #         x, y, w, h = cv.boundingRect(contour)
    #         cv.rectangle(frame, (x, y), (x+w, y+h), (250, 229, 202), 2)

    #     # Check if the center of the bounding box crosses the line
    #         center_y = y + h // 2
    #         if center_y > line_y - 5 and center_y < line_y + 5:  # Allow small tolerance
    #             print("Object crossed the line!")



    #showing effects
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.imshow('Edge Detected', edges)

    #exit
    key = cv.waitKey(30) & 0xFF
    if key == 27 or key == ord('q'):
        break

capture.release()
cv.destroyAllWindows()

#i could add event detection -- adding a line to cross that creates sound or takes a picture etc.
#causes a robotic output from motion detected - would the edge detection be for show

