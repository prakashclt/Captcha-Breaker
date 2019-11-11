
import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from helpers import resize_to_fit
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

def printimg(image1):
    plt.imshow(image1)

data = []
labels = []

LETTER_IMAGES_FOLDER = "letter_images"

# loop over the input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
#    printimg(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    printimg(image)
    image = resize_to_fit(image, 20, 20)


    # name of image
    label = image_file.split(os.path.sep)[-2]

    data.append(image)
    labels.append(label)
plt.imshow(data[5].reshape(20,20))
for i in range(len(data)):
    data[i]=data[i].flatten()   

testdata = data[:4]
data = data[4:]
labels = labels[4:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.25) 



from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
clf_lr = LogisticRegression(random_state=0 , solver = 'lbfgs',multi_class='multinomial')
clf_lr.fit(X_train,y_train)
pred = clf_lr.predict(X_test)
print(metrics.f1_score(y_test,pred,average='micro')) #0.9930392616924761
print(metrics.classification_report(y_test,pred))
filename = 'finalized_model.sav'
joblib.dump(clf_lr, filename)


from sklearn.naive_bayes import MultinomialNB
clf_NB = MultinomialNB()
clf_NB.fit(X_train,y_train)
pred = clf_NB.predict(X_test)
print(metrics.f1_score(y_test,pred,average='micro')) #0.9412638823713436
filename = 'finalized_model1.sav'
joblib.dump(clf_NB, filename)



k = [3,5,7,9]

from sklearn.neighbors import KNeighborsClassifier
for i in k:
    clf_knn = KNeighborsClassifier(n_neighbors=int(i))
    clf_knn.fit(X_train,y_train)
    pred = clf_knn.predict(X_test)
    print(metrics.f1_score(y_test,pred,average='micro'),i)

for i in range(5):
    print("True:",y_test[i])
    print("Pred:",pred[i])    
    printimg(X_test[i].reshape(20,20))
    


from sklearn.svm import SVC
model = SVC(gamma='auto')
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(metrics.f1_score(y_test,pred,average='micro'),i)

