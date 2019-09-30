import os
import sys
import glob
from math import ceil
from PIL import Image , ImageFilter , ImageDraw
import dlib
import time
import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans
from findplate import findplate
from skimage.filters import sobel

# In this example we are going to train a plate  detector based on the small
# plate dataset in the car_plates/ directory.  This means you need to supply
# the path to this plates folder as a command line argument so we will know
# where it is.
def splitcharacter(imagein,find_plate):
    img_arr2=dlib.as_grayscale(imagein)
    img_arr2=dlib.threshold_image(img_arr2)
    #print(type(imagein))
    #plate_str=find_plate.get_platestr_from_image(img_arr2)
    #print(plate_str)
    #img1=Image.fromarray(img_arr2,'L')
    '''rects = []
    dlib.find_candidate_object_locations(img_arr2, rects, min_size=50)
    print(len(rects))
    mywin = dlib.image_window()
    for rect in rects:
        #img2=img1
        #imagefinal=ImageDraw.Draw(img2)
        #imagefinal.rectangle(((rect.left(),rect.top()),(rect.right(),rect.bottom())),outline="black")
        #img2.show()
        mywin.clear_overlay()
        mywin.set_image(img_arr2)
        mywin.add_overlay(rect)
        #dlib.hit_enter_to_continue()
        time.sleep(0.1)
        #print(rect)
        '''
    #--------------------------------------------
    xmargin=ceil(float(lendiff)/8)
    ymargin=ceil(float(hdiff*9)/2)
    img=255-img_arr2
    h = img.shape[0]
    w = img.shape[1]

    print(h,w,"<==")
    #find blank columns:
    white_num = []
    white_max = 0
    for i in range(w):
        white = 0
        for j in range(h):
            #print(img[j,i])
            if img[j,i] <127:
                white += 1
        white_num.append(white)
        white_max = max(white_max, white)
    blank = []
    print("whitre_max=%d"%white_max)
    for i in range(w):
        if (white_num[i]  > 0.90 * white_max):
            blank.append(True)
        else:
            blank.append(False)

    #split index:
    i = 0
    num = 0
    l = 0
    x,y,d = [],[],[]
    while (i < w):
        if blank[i]:
            i += 1
        else:
            j = i
            while (j<w)and((not blank[j])or(j-i<10)):
                j += 1
            x.append(i)
            y.append(j)
            d.append(j-i)
            l += 1
            i = j
    print("len=%d"%l)
    failbox=[]
    whitesum=0
    for k in range(l):
        for i in range(x[k],y[k]):
            whitesum += white_num[i]
        failbox.append((100*whitesum)/(h*(y[k]-x[k])))
        #if ((100*whitesum)/(h*(y[k]-x[k]))) < 20 :
        #    failbox.append(True)
        #else:
        #    failbox.append(False)
        whitesum=0


    avgdiff2=round(w/8)
    avgdiff=0
    sumavg=1
    for k in range(l):
        print(x[k],y[k],d[k],avgdiff,sumavg)
        if (d[k]/avgdiff2)<1: #xmargin):
            avgdiff=avgdiff+d[k]
            avgdiff2=d[k]
            sumavg=sumavg+1
    avgdiff=round(avgdiff/sumavg) +2*xmargin
    """
    for k in range(l):
        if k==0 :
            print(x[k],y[k],d[k])
        else:
            print(x[k],y[k],d[k],round(d[k]/(avgdiff/(k+1))))
        if k==0 or round(d[k]/(avgdiff/(k+1)))>1:
            avgdiff = avgdiff + d[k]
    avgdiff=round((avgdiff)/l)-xmargin
    """
    print("*(%d)*"%avgdiff)
    for k in range(l):
        if round((d[k]*1.0)/avgdiff)>1 and l>=8:
            if k==0:
                x[k]=x[k]+avgdiff
            elif k==l-1:
                y[k]=y[k]-avgdiff
                if y[k]-x[k]< avgdiff :
                    y[k]=x[k]+avgdiff+xmargin
        if round((d[k]*1.0)/avgdiff)<1 :
            failbox[k]=2
        print(x[k],y[k],round((d[k]*1.0)/avgdiff))
        #if y[k]-x[k]< avgdiff :
        #    y[k]=x[k]+avgdiff
    print(failbox)

    realidx=0
    while l > 8:
        for k in range(len(failbox)):
            if failbox[k] < 25:
                del x[realidx]
                del y[realidx]
                del d[realidx]
                failbox[k]=-1
                l= l-1
            else:
                realidx+=1
        k=0
        lk=len(failbox)
        while k<lk:
            if failbox[k]==-1:
                del failbox[k]
                lk = lk-1
            else:
                k = k+1
        print(failbox)
        for ifl in range(8,l):
            doval = min(failbox)
            doval_idx= failbox.index(doval)  
            del x[doval_idx],y[doval_idx],d[doval_idx],failbox[doval_idx]
            l = l-1

    realidx=0
    for k in range(len(failbox)):
        if k >= len(failbox) :
            break
        
        if failbox[k] < 20:
            del x[realidx]
            del y[realidx]
            del d[realidx]
            del failbox[k]
            l= l-1
        else:
            realidx+=1
 
    if l< 8:
        k=0
        while k<l:
            print(k,d[k]/avgdiff,x[k],y[k],d[k])
            if d[k]/avgdiff>1.80:
                dn=d[k]-avgdiff
                d[k]=avgdiff
                yn=y[k]
                y[k]=x[k]+avgdiff
                if k==5:
                    xn=x[k]+avgdiff+xmargin*3
                elif k==2:
                    d[k]=avgdiff*2;
                    y[k]=x[k]+2*avgdiff
                    xn=x[k]+avgdiff*2-xmargin
                else:
                    xn=x[k]+avgdiff+xmargin
                #k=k+1
                if yn<= xn :
                    k=k+1
                    continue
                x.insert(k+1,xn)
                y.insert(k+1,yn)
                d.insert(k+1,dn)
                print(xn,yn,dn)
                l=l+1
                if l==8:
                    break
            k=k+1
    print("--------%d---------------------------------"%l)
    #--------------------------------------------
    #print (type(img_arr2))
    #print(" ")
    #print(img_arr2)
    #img_arr2=sobel(img_arr2)
    img1=Image.fromarray(img_arr2)
    #img1=img1.filter(ImageFilter.CONTOUR)
    #img1=img1.convert('1')
    #img1.show()
    img1.save('tmp.jpg')
    img2=dlib.load_rgb_image('tmp.jpg')
    ximg,yimg=img1.size
    print(img1.size,img2.shape)
    img2=img2[2*int(yimg/4):3*int(yimg/4),0:ximg]
    #img2=np.asarray(img1,dtype='int32')
    #img22=Image.fromarray(img2)
    #img22.show()
    img_arr=dlib.as_grayscale(img2)
    img_arr=dlib.threshold_image(img_arr)
    #print (type(img2))
    #print(" ")
    #print(img_arr)
    Data= {'x':[],'y':[]}
    for y2 in range(len(img_arr)):
        for x2 in range(len(img_arr[0])):
            if img_arr[y2][x2]<128:
                Data['x'].append(x2)
                Data['y'].append(y2)

    df = DataFrame(Data,columns=['x','y'])
    cluster=10
    try:
        kmeans = KMeans(n_clusters=cluster).fit(df)
    except Exception as e:
        return Null
    centroids=kmeans.cluster_centers_

    #print(len(centroids),centroids)
    centroids=sorted(centroids,key = lambda x2: x2[0])
    #centroids.reverse();
    print(centroids)
    imageofnumber=[]
    print ("==>",xmargin,ymargin)
    imagefinal=ImageDraw.Draw(img1)
    #for point in centroids:
        #imagefinal.ellipse((point[0]-int(avgdiff/2),point[1]+int(yimg/3)-2,point[0]+int(avgdiff/2),point[1]+int(yimg/3)+2),fill=55)
        #imagefinal.rectangle(((point[0]-int(avgdiff/2),0),(point[0]+int(avgdiff/2),yimg-3)),outline="blue")
        #imagefinal.rectangle(((point[0]-xmargin,point[1]-ymargin),(point[0]+xmargin,point[1]+ymargin)),outline="black")
    for k in range(l):
        imagefinal.rectangle(((x[k],1),(y[k],yimg-1)),outline="green")
    #xwalk=int(ximg-5)/10
    #ywalk=int(yimg-4)
    #xx=xwalk
    
    #while( xx < ximg-xwalk ):
    #    imagefinal.rectangle(((xx,2),(xx+xwalk,2+ywalk)),outline="black")
    #    xx=xx+xwalk
    img1.show() 
    firstx=0
    xstep=0
    """
    for point in centroids:
        if firstx==0:
            firstx=point[0]-int(avgdiff/2)
        else:
            #print(int(firstx-xmargin),int(0),int(point[0]+xmargin),int(yimg))
            print(int(firstx-avgdiff/2),int(0),int(point[0]+avgdiff/2),int(yimg))
            p11=int(0)#point[1]-ymargin)
            p12=int(yimg)#point[1]+ymargin)
            #p21=int(firstx-xmargin)#point[0]-xmargin)
            p21=int(firstx-avgdiff/2)#point[0]-xmargin)
            if xstep==0:
                xstep=point[0]-firstx
            p22=int(point[0]+avgdiff/2)
            if p11 < 0 :
                p11=0
            if p21 < 0:
                p21=0
            firstx=point[0]-int(avgdiff/2)
            imageofnumber.append(imagein[p11:p12,p21:p22])

    """ 
    for i in range(l):
        print(x[i],y[i])
        if x[i]<3 :
            x[i]=3
        if y[i]>w-2 :
            y[i]=w-2
        
        imageofnumber.append(imagein[0:int(yimg),int(x[i]-2):int(y[i])+2])
    return  imageofnumber

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Give the path to the car plates directory as the argument to this "
            "program."
            "execute this program by running:\n"
            "    ./read_car_plates_dlib.py car_plates/")
        plate_folder = "car_plates/"
        #exit()
    else:
        plate_folder = sys.argv[1]
    
    find_plate = findplate()
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
    file_list=[]
    if  plate_folder.find(".jpg",-4)>-1 :
        file_list.append(plate_folder)
    else:
        file_list=glob.glob(os.path.join(plate_folder, "*.jpg"))

    for f in file_list:#glob.glob(os.path.join(plate_folder, "*.jpg")):
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
            charsplited=splitcharacter(imgnew,find_plate)
            cnt=0
            for isp in charsplited:
                win.set_image(isp)
                #print(isp)
                imgs=Image.fromarray(isp)
                imgs.save(str(cnt)+".jpg")
                cnt+=1
                #print(find_plate.get_platestr_from_image(isp))
                #time.sleep(1)
                #dlib.hit_enter_to_continue()
            imagess=[]
            for i in range(cnt):
                imgss=dlib.load_rgb_image(str(i)+".jpg")
                imagess.append(imgss)
            print(find_plate.get_platestr_from_image(imagess))#charsplited))
            #print(find_plate.get_platestr_from_image(charsplited))
            #for io in range(len(di2)):
            #    print(di2[io].shape)
            #    print(charsplited[io].shape)
            #    if cmp(charsplited[io].reshape(1,-1),di2[io].reshape(1,-1))==0:
            #    #if cmp([io]==di2[io]:
            #        print("the same...")
            #    else:
            #        print("diff in %d"%io)
            ####--------------------------------------------
            #myimg = p_image.crop((left,top,right,bottom))
            #myimg.show()
            #win.set_image(imgnew)
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

    print("finished.....")
    dlib.hit_enter_to_continue()
