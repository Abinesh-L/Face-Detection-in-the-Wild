# Face-Detection-in-the-Wild
In this project there are 2 tasks performed: 1) Face Detection: There are a set of images given where each image consists of people. The task is to identify the faces in the images and annotate them 2) Face Clustering: Images of different people are given. The task is to identify and cluster all the images of same identity using K-means, and do the same for all other people in the given images.
# TASK 1:
\n For this task, I used Haar feature based cascade classifiers to detect the faces
in the images.
Haar Feature:
Haar feature works based on the concept of Viola–Jones object detection
framework where the machine learning model, is trained using a cascade
function of positive and negative images. It is then used to detect faces in
other images.
A folder full of images is provided. The program is coded (FaceDetector.py),
so that it will be able to detect faces in the given images and store the
“box” (the highlighted area around a particular face) details in a json file,
namely results.json. The box details contain the x and y coordinate for the
top-left corner along with the width and height of the box, in that order.
An image element in my results.json file will look like so:
[{"iname": "img.jpg", "bbox": [x, y, width, height]}, ...]
# TASK 2:
I used the concept of K-Means Clustering on the 128-bin output of
face_recognition.face_encodings. The program is coded (FaceCluster.py), so
that it will be able to detect faces in the given images. Then this list is passed
to face_recognition.face_encodings to get the 128-bin feature output. In this
append vector of size(Nx128), where N is the number of images, the concept
of K-means clustering is applied to sort the images based on the feature. The
number of features used is 128. The final clustered output is below.
