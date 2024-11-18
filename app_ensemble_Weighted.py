# -*- coding: utf-8 -*-

from __future__ import division, print_function
#Import necessary libraries
# Flask utils
from flask import Flask, redirect, url_for, render_template, request
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

import numpy as np
import os
import sys
import glob
import re
import tensorflow as tf
print(tf.__version__)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
input = Input(shape=(224, 224, 3))
#model_outputs = [model(input) for model in models]

#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
#from keras.models import load_model

class WeightedAverageLayer(tf.keras.layers.Layer):
    def __init__(self, w1, w2, w3, w4,**kwargs):
        super(WeightedAverageLayer, self).__init__(**kwargs)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'w1': self.w1,
            'w2': self.w2,
            'w3': self.w3,
            'w4': self.w4,
        })
        return config
        
    def call(self,inputs):
        return self.w1 * inputs[0] + self.w2 * inputs[1] + self.w3 * inputs[2] + self.w4 * inputs[3]


ensemble_output = WeightedAverageLayer(0.3, 0.6, 0.1, 0.9) #(model_outputs)
#Loading the Model
# Model saved with Keras model.save()
MODEL_PATH ='model/ensemble_weighted_averaging94.h5'
# Load your trained model
#model = load_model(MODEL_PATH)
model = load_model(MODEL_PATH , custom_objects={'WeightedAverageLayer': WeightedAverageLayer})

print('@@ Model loaded')

#Prediction def model_predict(img_path, model):
def model_predict(img_path, model):
    print(img_path)
    test_img = image.load_img(img_path, target_size=(224, 224)) # load image 
    print("@@ Got Image for prediction")
    
    # Preprocessing the image
    x = image.img_to_array(test_img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    print('@@ Raw result = ', preds)
    preds = np.argmax(preds, axis=1)
    #preds = np.argmax(preds)
    if preds==0:
        preds="The Person maybe Suffering from Covid Pneumonia", 'Covid_Pneumonia.html'  # if index 0
    elif preds==1:
        preds="The Person maybe Normal", 'Normal.html'  # if index 1
    elif preds==2:
        preds="The Person maybe Suffering from  Pneumonia", 'Pneumonia.html'  # if index 2
    else:
        preds="The Person maybe Suffering from Tuberculosis Pneumonia", 'Tuberculosis_Pneumonia.html' # if index 3
        
    
    
    return preds


# Create flask instance
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        preds, output_page = model_predict(file_path, model)
        
        return render_template(output_page, pred_output = preds, user_image = file_path)
        """
        preds = model_predict(file_path, model)
        result=preds
        return result
     return None """
    
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,) 
    
    