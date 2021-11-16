import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential 

import cv2 
import os 
import glob
import sklearn
import numpy as np

from matplotlib import pyplot as plt 
from math import log10, sqrt
from sklearn import metrics
from skimage.measure import compare_ssim
import argparse
import imutils
from tensorflow.keras.preprocessing.image import ImageDataGenerator

Image read
img_dir = r"C:/Users/Surya/project/ dataset/original_img" 
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
data = [] 
for f1 in files: 
    img = cv2.imread(f1,1) 
    data.append(img) 

#APPLY CANNY
img_dir_c = r"C:/Users/Surya/project/ dataset/generated_img"
p=f'canny_img.jpg'
p=((img_dir_c+'/')+p)
cannylist=[]
p1=f'sobelx8u.jpg'
p1=((img_dir_c+'/')+p1)
sx8u=[]
s8u=[]

i=0
for f1 in data:
    edges_Canny = cv2.cv2.Canny(f1,100,200)
    cannylist.append(edges_Canny)

    #SAVE_IMAGE
    cv2.imwrite(p,cannylist[i])

    #READ_IMAGE
    cannylist_img = cv2.imread(r"C:/Users/Surya/project/ dataset/generated_img/canny_img.jpg",1) 

    a1,a2,a3=cannylist_img.shape
    cannylist_img=cannylist_img.reshape(a1,(a2*3))
    
    #SOBEL APPLIED-------------------------------------------
    sobelx8u = cv2.Sobel(f1,cv2.CV_8U,1,0,ksize=1)
    sobelx64f = cv2.Sobel(f1,cv2.CV_64F,1,0,ksize=1)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)

    sx8u.append(sobelx8u)
    s8u.append(sobel_8u)
    
    #SAVE_IMAGE
    cv2.imwrite(p1,sx8u[i])

    #READ_IMAGE
    sx8u_img = cv2.imread(r"C:/Users/Surya/project/ dataset/generated_img/sobelx8u.jpg",1) 

    #3D TO 2D
    b1,b2,b3=sx8u_img.shape
    sx8u_img=sx8u_img.reshape(b1,(b2*3))
    newarr=np.zeros((b1,b2*3))
    for j in range(a1):
        for k in range(a2):
            newarr[j][k]=cannylist_img[j][k]
    cannylist_img=newarr

    #CALCULATE ERRORS
    plt.subplot(1,3,1),plt.imshow(f1,cmap = 'binary')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,2),plt.imshow(edges_Canny,cmap = 'gray')
    plt.title('Canny Image'), plt.xticks([]), plt.yticks([])  


    plt.subplot(1,3,3),plt.imshow(sobelx8u,cmap = 'gray')
    plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([]) 
    plt.show()
    
    value = PSNR(sx8u_img, cannylist_img)
    print(f"PSNR value is {value} dB")
    print('Mean Absolute Error:', metrics.mean_absolute_error(sx8u_img, cannylist_img))
    (score, diff) = compare_ssim(sx8u_img, cannylist_img, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {} %".format(score*100))
    print("-------------------------------------------------------------------")
    print('\n')
    i+=1

train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, 
        rotation_range=30,  
        zoom_range = 0.15,
        shear_range = 0.2,    
        horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory( r'C:/Users/Surya/project/New Covid Images Dataset/Covid training',
                                                  target_size=(64,64),
                                                  batch_size=32,
                                                  class_mode='binary')

val_set = val_datagen.flow_from_directory( r'C:/Users/Surya/project/New Covid Images Dataset/covid testing',
                                          target_size=(64,64),
                                          batch_size=32,
                                          class_mode='binary')


cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#------------------------------ VGG19 ----------------------------------------------------
Base_model = tf.keras.applications.VGG19(input_shape=[64, 64, 3],
                                              include_top=False,
                                              weights='imagenet')
Base_model.trainable = True
model = tf.keras.Sequential([Base_model,
                             keras.layers.GlobalAveragePooling2D(),
                             keras.layers.Dense(64, activation='relu'),
                             keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#------------------------------ MobileNetV2 ----------------------------------------------------
Base_model = tf.keras.applications.MobileNetV2(input_shape=[64, 64, 3],
                                              include_top=False,
                                              weights='imagenet')
Base_model.trainable = False
model1 = tf.keras.Sequential([Base_model,
                             keras.layers.GlobalAveragePooling2D(),
                             keras.layers.Dense(64, activation='relu'),
                             keras.layers.Dense(1, activation='sigmoid')])

model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#------------------------------ LeNet ----------------------------------------------------
cnn2 = tf.keras.models.Sequential()

cnn2.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="tanh", input_shape=[64, 64, 3]))
cnn2.add(tf.keras.layers.BatchNormalization())
cnn2.add(tf.keras.layers.AvgPool2D(pool_size=2, strides=2, padding='valid'))
cnn2.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="tanh"))
cnn2.add(tf.keras.layers.AvgPool2D(pool_size=2, strides=2, padding='valid'))


cnn2.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="tanh"))

cnn2.add(tf.keras.layers.Flatten())

cnn2.add(tf.keras.layers.Dense(units=64, activation='relu'))
cnn2.add(tf.keras.layers.Dense(units=32, activation='relu'))

cnn2.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print("--------------------------------------------VGG---------------------------------------------------")
vgg = model.fit(
          training_set,
          steps_per_epoch=15,
          epochs=25,
          validation_data = val_set,
          validation_steps=10
         )
print("-------------------------------------------Mobile--------------------------------------------------")
mobile = model1.fit(
          training_set,
          steps_per_epoch=15,
          epochs=25, 
          validation_data = val_set,
          validation_steps=10
         )
print("---------------------------------------------CNN---------------------------------------------------")
history = cnn.fit(
          training_set,
          steps_per_epoch=15,
          epochs=25, 
          validation_data = val_set,
          validation_steps=10
        )
print("---------------------------------------------LeNet-----------------------------------------------")
history2 = cnn2.fit(
          training_set,
          steps_per_epoch=15,
          epochs=25, 
          validation_data = val_set,
          validation_steps=10
        )

epochs = 25

acc_le = history2.history['accuracy']
acc_cnn = history.history['accuracy']
acc_mob = mobile.history['accuracy']
acc_vgg = vgg.history['accuracy']
epochs_range = range(epochs)
plt.plot(epochs_range, acc_cnn, label='Training Accuracy CNN')
plt.plot(epochs_range, acc_le, label='Training Accuracy LeNet')
plt.plot(epochs_range, acc_vgg, label='Training Accuracy VGG')
plt.plot(epochs_range, acc_mob, label='Training Accuracy MobileNet')
plt.legend(loc='lower right')
plt.title('Training Accuracy')
plt.show()

epochs=25
val_acc_le = history2.history['val_accuracy']
val_acc_cnn =history.history['val_accuracy']
val_acc_mob =mobile.history['val_accuracy']
val_acc_vgg =vgg.history['val_accuracy']
epochs_range = range(epochs)
plt.plot(epochs_range, val_acc_cnn, label='Validation Accuracy CNN')
plt.plot(epochs_range, val_acc_le, label='Validation Accuracy LeNet')
plt.plot(epochs_range, val_acc_vgg, label='Validation Accuracy VGG')
plt.plot(epochs_range, val_acc_mob, label='Validation Accuracy MobileNet')
plt.legend(loc='lower right')
plt.title('Validation Accuracy')
plt.show()
epochs=25
loss_le =history2.history['loss']
loss_cnn =history.history['loss']
loss_mob =mobile.history['loss']
loss_vgg =vgg.history['loss']
epochs_range = range(epochs)
plt.plot(epochs_range, loss_cnn, label='Training loss CNN')
plt.plot(epochs_range, loss_le, label='Training loss LeNet')
plt.plot(epochs_range, loss_vgg, label='Training loss VGG')
plt.plot(epochs_range, loss_mob, label='Training loss MobileNet')
plt.legend(loc='center')
plt.title('Training Loss')
plt.show()
epochs=25
val_loss_le =history2.history['val_loss']
val_loss_cnn =history.history['val_loss']
val_loss_mob =mobile.history['val_loss']
val_loss_vgg =vgg.history['val_loss']
epochs_range = range(epochs)
plt.plot(epochs_range, val_loss_cnn, label='Validation loss CNN')
plt.plot(epochs_range, val_loss_le, label='Validation loss LeNet')
plt.plot(epochs_range, val_loss_vgg, label='Validation loss VGG')
plt.plot(epochs_range, val_loss_mob, label='Validation loss MobileNet')
plt.legend(loc='center')
plt.title('Validation Loss')
plt.show()
import numpy as np
from keras.preprocessing import image
p=r"C:/Users/Surya/project/New Covid Images Dataset/covid testing/Covids/3cbfec 1259635898efae5b57ea3ddea3.jpg"
test_image = image.load_img(p, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image/=255.0
result = cnn.predict(test_image)
print("---- By CNN ----")
if result<0.5:
  print('Covid Detected') 
else:
  print('Not suffering from Covid')
result_vgg = model.predict(test_image)
print("---- By VGG ----")
if result_vgg<0.5:
  print('Covid Detected') 
else:
  print('Not suffering from Covid')
result_mobile = model1.predict(test_image)
print("---- By MobileNet ----")
if result_mobile<0.5:
  print('Covid Detected') 
else:
  print('Not suffering from Covid')
result_lenet = cnn2.predict(test_image)
print("---- By LeNet ----")
if result_lenet<0.5:
  print('Covid Detected') 
else:
  print('Not suffering from Covid')
def show(pred,img):
  if pred<0.5: str = 'Covid Detected'
  else: str = 'Not suffering from Covid'
  plt.imshow(img)
  plt.axis('on')
  plt.title(str)
  plt.show()
img = image.load_img(p)
print("------ CNN ------")
show(result,img)
print("------ VGG-19 ------")
show(result_vgg,img)
print("------ MobileNet_V2 ------")
show(result_mobile,img)
print("------ LeNet ------")
show(result_lenet,img)
