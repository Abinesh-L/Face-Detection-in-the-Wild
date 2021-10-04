# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
import face_recognition
import sys

   

def KMeans(centroids, fvect, n_cluster):        
    C = np.vstack(np.array(centroids))
    cluster = np.zeros(len(fvect))
    n_clusters = np.arange(n_cluster)
    centerlistnew = list()
    clusterlist = list()
    #C = np.copy(cent)
    D = np.vstack(fvect)
    for k in range(iter):
         #print(C)
         #prevC = C
         #imagen = []
         for i in range(len(D)):
             dis =  C - D[i]         
             cluster[i] = np.argmin(np.sqrt(np.square(dis).sum(axis=1)),axis=0)
         clusterlist.append(cluster)
             
         for j in range(n_cluster):
             W = np.array(np.where(cluster == n_clusters[j]))        
             if not all(map(lambda x: all(x), W)):  #To access all values of W
               C[j,:] = np.average(D[W,:], axis=1)
             #imagen.append(W) 
         centerlistnew.append(C)
         if k>1 and centerlistnew[k].all() == centerlistnew[k-1].all():
             break
    return(cluster)
    
# Centroid Initialization
def initial(vectdata, n):
    centroids = list()
    centroids.append(vectdata[np.random.randint(vectdata.shape[0]), :])
    for cent in range(n - 1): 
        d = list()
        for i in range(vectdata.shape[0]):
            point = vectdata[i, :]
            max = sys.maxsize              
            for j in range(len(centroids)):
                temp = np.sum((point - centroids[j])**2)
                max = min(max, temp)
            d.append(max)
              
        ## data point with max distance as next centroid
        d = np.array(d)
        next_centroid = vectdata[np.argmax(d), :]
        centroids.append(next_centroid)
    return centroids

def createjson(cluster_data, n_cluster):
    clno, imname, img_final = list(zip(*cluster_data))
    cluster_final = np.array(list(zip(clno, imname)))
    n_clusters = np.arange(n_cluster).astype(str)
    json_list = []
    for j in range(n_cluster):
        index = np.array(np.where(cluster_final[:,0]== n_clusters[j]))
        clist = (cluster_final[index,1]).tolist()
        clusterdata={"cluster_no":n_clusters[j], "elements" :clist}
        json_list.append(clusterdata)
        
#To get the clustered images remove the comments and run the code
# =============================================================================
#         imlist = []
#         for k in range(len(index[0])):
#             img_box = img_final[index[0][k]]
#             img_box = cv2.resize(img_box, (100,100), interpolation=cv2.INTER_LINEAR)
#             imlist.append(img_box)
#         imgcl = np.hstack(imlist)
#         cv2.imwrite('cluster'+n_clusters[j]+'.jpg', imgcl)
#         plt.imshow(imgcl)
# =============================================================================

    return(json_list)


if __name__ == "__main__":

    #ele_list = [] 
    #elementdata = []
    fvector_list = []
    files = []
    imgdata = [] 
    imgbox = []
    imgpath = ""

    cmd_args = sys.argv

    args = sys.argv

    for i in range(1, len(args)):
        if i == 1:
            imgpath = imgpath + args[i]
        else:
            imgpath = imgpath + " " + args[i]

    #img_path = os.path.join(img_path, "images")
    
    #Get number of clusters from folder name
    #filename = ([filename for filename in os.listdir(os.getcwd()) if filename.startswith('faceCluster_')])
    #n_cluster = int(filename[0][-1])
    #F = sys.argv
    K = int(imgpath[-1])
    n_cluster = K
    #Read all the images
    for image in os.listdir(imgpath):        
        img = cv2.imread(imgpath + "\\" + image)
        img1 = np.copy(img)
        imgdata.append(img)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_frontalface_alt2.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)
        for (x,y,w,h) in faces:
            #elementdata={"iname":image, "bbox" :[float(x), float(y), float(w), float(h)]}
            #ele_list.append(elementdata)
            boximg = img[y:y+h, x:x+w]
            box = [(y,x+w,y+h,x)]
            fvector = face_recognition.face_encodings(img,box)
            fvector_list.append(np.array(fvector))
            files.append(image)
            imgbox.append(boximg)
    fvect = np.array(fvector_list).reshape(len(files),128) #Face_encodings vector
    #imgbox = np.array(imgbox)
    imgdata = np.array(imgdata)
    iter = 100
    
    cluster = np.zeros(len(fvect))
    n_clusters = np.arange(n_cluster)
    centroids = initial(fvect, n_cluster) #Centroid initialization
    
    
    cluster = KMeans(centroids, fvect, n_cluster) #KMeans Implementation
    cluster = np.array(cluster).astype(int)
    n_clusters = np.array(n_clusters).astype(str)
    cluster_data = list(zip(cluster, files, imgbox))
    #Clustered image data list
    cluster_data.sort()
    
    json_list = createjson(cluster_data, n_cluster)
    output_json = (os.getcwd()+".\\cluster.json")
    with open(output_json, 'w') as f:
        json.dump(json_list, f)



    


