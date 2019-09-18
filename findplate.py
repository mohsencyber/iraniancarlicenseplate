from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn import datasets , svm , metrics
from PIL import Image , ImageDraw
from PIL import ImageFilter
import numpy as np
import os
import dlib
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import sobel

class findplate:
    def __init__(self):
        print("inited.......")
    
    #def __read_image(self,images):
        ################custom train
        dataimage=[]
        target=[]
        train_imgs=os.listdir('train/')
        cnt=0
        self.dict_label={}
        size=28,42
        for train_img in train_imgs:
            #print(train_img)
            img_label=train_img.split('.')[0]
            img_label=img_label[:-1]
            #if not img_label.isdigit():
            #    continue
            img_label=img_label.rjust(3,' ')
            try:
                imgg=dlib.load_rgb_image("train/"+train_img)
                imgg=dlib.as_grayscale(imgg)
                #imgg,rect=dlib.gaussian_blur(imgg,max_size=2000L)
                imgg=dlib.threshold_image(imgg)
                #img=Image.open("train/"+train_img)
                img=Image.fromarray(imgg)
                img=img.resize((30,43), Image.ANTIALIAS)
                #img=img.filter(ImageFilter.GaussianBlur())
                #data_img=rgb2gray(io.imread("train/"+train_img,as_gray=True))
                #data_img=sobel(data_img)
                img=img.convert('L')
                #img.show()
                data_img=np.asarray(img,dtype="int32")
                data_img=np.concatenate(data_img,axis=0)
                #print(data_img)
                print("-------------------"+train_img)
                #print(img_label)
                cnt +=1
                cntp=cnt
                if not self.dict_label.has_key(img_label):
                    self.dict_label[img_label]=cntp
                else:
                    cntp=self.dict_label[img_label]
                dataimage.append(data_img)
                #print(cntp)
                #target.append(cntp)
                target.append(img_label)
            except IOError:
                pass
        
        print("-------------------",len(dataimage))
        #self.classifier = svm.SVC(probability=True,  kernel="rbf", C=2.8, gamma=.00073,verbose=10)#gamma=0.001 , C=100.)
        self.classifier = svm.SVC(gamma=0.000001 , C=100. )
        self.classifier.fit(dataimage, target)
        y_hat = self.classifier.predict(dataimage)#[dataimage[3],dataimage[15]])
        #print(y_hat)
        #print(target[3],target[15])
        acc = np.mean(y_hat == target)
        print("\n\nTraining Accuracy for  %.2f\n"%(acc))        
       

    def __read_image(self,images):
        dataimage=[]
        for image in images:
            #mage=Image.open('test.jpg')
            #image=image.rotate(180)
            #crope=(30,10,155,80)
            #image=image.crop(crope)
            image=dlib.as_grayscale(image)
            image=dlib.threshold_image(image)
            #img_arr=np.asarray(image,dtype='int32')
            img_int=Image.fromarray(image)
            img_int=img_int.resize((30,43),Image.ANTIALIAS)#30,43
            img_int=img_int.convert('L')
    
            #img_int.show()
            #img_name=str(raw_input("enter image name this:"))
            #print(type(img_name))
            #imagename="train/"+str(img_name)+".jpg"
            #if not os.path.isfile(imagename):
            #    img_int.save("train/"+img_name+".jpg")
            #else:
            #    print("image exists.")
            img_arr=np.asarray( img_int,dtype="int32")
            img_arr=np.concatenate(img_arr,axis=0)
            #print(img_arr)
            #imagefinal=ImageDraw.Draw(image)
            #for point in centroids:
                #imagefinal.rectangle(((point[0]-15,point[1]-20),(point[0]+15,point[1]+23)),outline="black")
            #image.show()
            #print(img_arr)
            dataimage.append(img_arr)
            
        #print(dataimage[0].shape)
        predicted = self.classifier.predict(dataimage)#img_arr)
        self.number_string=''
        print(predicted)
        for i in predicted:
            self.number_string+=i.strip()
            #print(i)
        ############################

    def get_platestr_from_image(self,src_img):
        self.__read_image(src_img)
        return self.number_string
