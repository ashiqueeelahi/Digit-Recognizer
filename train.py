import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression ;
from sklearn.linear_model import Ridge, Lasso;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.naive_bayes import GaussianNB;
import xgboost as xgb;
from xgboost import XGBClassifier;
from xgboost import XGBRegressor;

import tensorflow as tf
import keras;
from keras_preprocessing import image;
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam;
from keras.callbacks import ModelCheckpoint;
from keras.models import Sequential;
from tensorflow.keras.applications import VGG16;
from tensorflow.keras.applications import InceptionResNetV2;
from keras.applications.vgg16 import preprocess_input;
from tensorflow.keras.applications.vgg16 import decode_predictions;
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import os;
from os import listdir;
from PIL import Image as PImage;
import cv2

train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")

data_gen = ImageDataGenerator(rescale= 1/255,
                              rotation_range = 40,
                              width_shift_range = 0.2,
                              height_shift_range=0.2, 
                              shear_range=0.2, 
                              zoom_range = 0.2,
                              horizontal_flip= True,
                              fill_mode = 'nearest'
                              )

test_gen = ImageDataGenerator(rescale= 1/255)

x_train = train.drop('label' , axis = 1);
y_train = train['label']

X_train,X_test,y_train,y_test = train_test_split(x_train, y_train,test_size=0.15,random_state=11)

X_train.shape , y_train.shape

X_train = X_train.values.reshape(len(X_train),28,28,1)

X_test = X_test.values.reshape(len(X_test),28,28,1)

training_data = data_gen.flow(X_train,y_train,
                             #target_size = (150,150),
                             batch_size = 32, 
                             #class_mode = 'binary'
                              #seed = 11
                             )

testing_data = data_gen.flow(X_test,y_test,
                             #target_size = (150,150),
                             batch_size = 32) 

model = keras.models.Sequential([
                        keras.layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu', input_shape = (28,28,1)),
                        keras.layers.MaxPooling2D(pool_size=(2,2)),
                        keras.layers.Conv2D(filters = 64, kernel_size = 3),
                        keras.layers.MaxPooling2D(pool_size= (2,2)),
                        keras.layers.Conv2D(filters = 128, kernel_size = 3),
                        keras.layers.MaxPooling2D(pool_size= (2,2)),
                        keras.layers.Conv2D(filters = 256, kernel_size = 3),
                        keras.layers.MaxPooling2D(pool_size= (2,2)),
                        
                        keras.layers.Dropout(0.50),
                        keras.layers.Flatten(),
                        keras.layers.Dense(units = 128, activation = 'relu'),
                        keras.layers.Dropout(0.10),
                        keras.layers.Dense(units = 256, activation = 'relu'),
                        keras.layers.Dropout(0.25),
                        keras.layers.Dense(units = 10, activation = 'softmax')
                        ])


cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,data_format='channels_last',activation='relu',input_shape=(28,28,1),padding="same"))
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,data_format='channels_last',activation='relu',padding="same"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) 
cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,data_format='channels_last',activation='relu',padding="same"))
cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,data_format='channels_last',activation='relu',padding="same"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,data_format='channels_last',activation='relu',padding="same"))
cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,strides=1,data_format='channels_last',activation='relu',padding="same"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
cnn.add(tf.keras.layers.Flatten())


cnn.add(tf.keras.layers.Dense(512,activation="relu"))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Dense(512,activation="relu"))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Dense(1024,activation="relu"))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Dense(10,activation="softmax"))

cnn.compile(optimizer= Adam(lr=0.0001), loss = 'sparse_categorical_crossentropy' , metrics= ['accuracy'])

early_stop = EarlyStopping( monitor="val_loss",
    min_delta=0,
    patience=0,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

 

cnn.fit_generator(training_data, epochs= 500, verbose = 1, callbacks=[early_stop], validation_data= testing_data)

cnn.evaluate(X_test, y_test)

test = pd.read_csv("../input/digit-recognizer/test.csv")

test1 = test/255.0

test2 = test1.values.reshape(len(test),28,28,1)

pred = cnn.predict_classes(test2)



final_submission = pd.DataFrame({"ImageId":range(1,len(pred)+1),"Label":pred})

final_submission.to_csv("final_submission_5.csv",index=False)

