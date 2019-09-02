import os
import sys
import glob
from math import ceil
from PIL import Image , ImageFilter
import dlib
import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans

# In this example we are going to train a plate  detector based on the small
# plate dataset in the car_plates/ directory.  This means you need to supply
# the path to this plates folder as a command line argument so we will know
# where it is.
def splitcharacter(imagein):
    img_arr2=dlib.as_grayscale(imagein)
    print (type(img_arr2))
    print(" ")
    print(img_arr2)
    img1=Image.fromarray(imagein,'RGB')
    img1=img1.filter(ImageFilter.CONTOUR)
    img1.show()
    img1.save('tmp.jpg')
    img2=dlib.load_rgb_image('tmp.jpg')
    #img2=np.asarray(img1,dtype='int32')
    img_arr=dlib.as_grayscale(img2)
    print (type(img2))
    print(" ")
    print(img_arr)
    Data= {'x':[],'y':[]}
    for y in range(len(img_arr)):
        for x in range(len(img_arr[0])):
            if img_arr[y][x]<=220:
                Data['x'].append(x)
                Data['y'].append(y)

    df = DataFrame(Data,columns=['x','y'])
    cluster=9
    kmeans = KMeans(n_clusters=cluster).fit(df)
    centroids=kmeans.cluster_centers_
    #print(len(centroids),centroids[2][0],centroids[2][1])

    centroids=sorted(centroids,key = lambda x: x[0])
    #centroids.reverse();
    print(centroids)
    imageofnumber=[]
    xmargin=ceil(float(lendiff)/2)
    ymargin=ceil(float(hdiff*9)/2)
    print (xmargin,ymargin)
    for point in centroids:
        print(int(point[0]-xmargin),int(point[0]+xmargin),int(point[1]-ymargin),int(point[1]+ymargin))
        p11=int(point[1]-ymargin)
        p12=int(point[1]+ymargin)
        p21=int(point[0]-xmargin)
        p22=int(point[0]+xmargin)
        if p11 < 0 :
            p11=0
        if p21 < 0:
            p21=0
        imageofnumber.append(imagein[p11:p12,p21:p22])
    return  imageofnumber

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Give the path to the car plates directory as the argument to this "
            "program."
            "execute this program by running:\n"
            "    ./read_car_plates_dlib.py car_plate/")
        exit()
    plate_folder = sys.argv[1]
    
    
    options = dlib.simple_object_detector_training_options()
    
    options.add_left_right_image_flips = True
    
    options.C = 50
    options.epsilon=0.00001
    options.num_threads = 4
    options.be_verbose = True
    
    training_xml_path = os.path.join(plate_folder, "training.xml")
    testing_xml_path = os.path.join(plate_folder, "testing.xml")
    
    print(options)
    #dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)
    
    print("")  # Print blank line to create gap from previous output
    #print("Training accuracy: {}".format(
    #    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
    #
    
    #print("Testing accuracy: {}".format(
    #    dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))
    
    detector = dlib.simple_object_detector("detector.svm")
    
    # We can look at the HOG filter we learned.  It should look like a plate.  Neat!
    #win_det = dlib.image_window()
    #win_det.set_image(detector)
    
    print("Showing detections on the images in the plates folder...")
    win = dlib.image_window()
    global lendiff , hdiff
    #wins=[]
    for f in glob.glob(os.path.join(plate_folder, "*.jpg")):
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)
        #print (img)
        #p_image=Image.fromarray(img,'RGB')
        #rects = []
        #dlib.find_candidate_object_locations(img, rects, min_size=100)
        #p_image=p_image.convert('L',dither=Image.NONE)
        #p_image=p_image.filter(ImageFilter.GaussianBlur())
        #p_image=p_image.filter(ImageFilter.CONTOUR)    
        #p_image=p_image.convert('1')#,dither=Image.NONE)
        #p_image.show()
        #print(np.asarray(p_image,dtype="uint8"))
        #dets = detector(np.asarray( p_image,dtype="uint8"))
        dets = detector(img)
        print("Number of plate detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            lendiff=(d.right()-d.left())/11
            hdiff=(d.bottom()-d.top())/9
            top = d.top()-hdiff
            bottom=d.bottom()+hdiff
            left= d.left()+lendiff/2
            right=d.right()+lendiff
            if  top < 0:
                top = 0
            if left < 0:
                left = 0
            imgnew=img[top:bottom,left:right]
            ####---------------Find Character---------------
            charsplited=splitcharacter(imgnew)
            for isp in charsplited:
                win.set_image(isp)
                dlib.hit_enter_to_continue()
            ####--------------------------------------------
            #myimg = p_image.crop((left,top,right,bottom))
            #myimg.show()
            win.set_image(imgnew)
            #wins.append(imgnew)
            dlib.hit_enter_to_continue()
        #win.clear_overlay()
        #win.set_image(img)
        #win.add_overlay(dets)
        #for i in wins:
            #inwin=dlib.image_window()
            #inwin.set_image(i)
            #dlib.hit_enter_to_continue()
        #wins *= 0
        if len(dets)==0:
            dlib.hit_enter_to_continue()


