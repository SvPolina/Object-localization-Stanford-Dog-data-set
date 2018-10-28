# Object-localization-Stanford-Dog-data-set

Object localization with Stanford Dog Dataset

Data set: Stanford Dog Dataset contains ~20 k images belonging to 120 classes which makes it ~ 180 images per class.
Though for object classification task this fact might be a serious drawback (small amount of information for each class), 
since our aim is to learn to localize a frame around the dog object the above fact is not a serious obstacle for our learning process.
We've doqnloaded the images while each image is accompanied with an annotation .html like file. 
This file contains informationabount the image name, it's size and (Xmin,Ymin) & (Xmax,Ymax) pairs corresponding to each 
frame that surrounds a dog in a given image. As a fist step in our data preprocessing we created 2 files 1-images.txt: each entry in
the file containing a partial path to given image 2-bounding_boxes.txt: each entry corresponds to a given image. 
The first 2 numbers are the Xmin,Ymin coordinate while the net 2 numbers correspond to the width and the height of the bounding box.

Learning process: 
For this task we've used the pretrained ResNet18. The lower layers of the network are activated with the primitive features 
(texture,color etc.) which makes them sharable with other applications. We've changed only the number of the learned 
classes to 4 - Xmin,Ymin,w,h of the bounding box.
After 10 epochs the achieved accuracy on the train dataset was:91.57%
While on the test dataset the achieved performance was:80%
