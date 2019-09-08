import cv2

### load input image and convert it to grayscale
img = cv2.imread("tmp.jpg")
print("img shape=", img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#### extract all contours
_, contours, _  = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# debug: draw all contours
#cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
#cv2.imwrite("all_contours.jpg", img)

#### create one bounding box for every contour found
bb_list = []
for c in contours:  
    bb = cv2.boundingRect(c)
    # save all boxes except the one that has the exact dimensions of the image (x, y, width, height)
    if (bb[0] == 0 and bb[1] == 0 and bb[2] == img.shape[1] and bb[3] == img.shape[0]):
        continue
    bb_list.append(bb)

# debug: draw boxes
img_boxes = img.copy()
for bb in bb_list:
   x,y,w,h = bb
   cv2.rectangle(img_boxes, (x, y), (x+w, y+h), (0, 0, 255), 2)
cv2.imwrite("boxes.jpg", img_boxes)    

#### sort bounding boxes by the X value: first item is the left-most box
bb_list.sort(key=lambda x:x[0])

# debug: draw the last box of the list (letter M)

#print("letter M @ ", bb_list[-1])
#x,y,w,h = bb_list[-1]
#cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
#cv2.imwrite("last_contour.jpg", img)

### remove the last item from the list, i.e. remove box for letter M
bb_list = bb_list[:-1]

### and now the fun part: create one large bounding box to rule them all
x_start, _, _, _ = bb_list[0]
x_end, _, w_end, _ = bb_list[-1]

x = x_start
w = (x_end + w_end) - x_start

bb_list.sort(key=lambda y:y[1]) # sort by Y value: the first item has the smallest Y value 
_, y, _, _ = bb_list[0]

bb_list.sort(key=lambda y:y[3]) # sort by Height value: the last item has the largest Height value 
_, _, _, h = bb_list[-1]

print("x=", x, "y=", y, "w=", w, "h=", h)

# debug: draw the final region of interest
roi_img = img.copy()
cv2.rectangle(roi_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
cv2.imwrite("roi.jpg", roi_img)

# crop to the roi
crop_img = img[y:y+h, x:x+w]
cv2.imwrite("crop.jpg", crop_img)
