import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

dirname = os.path.dirname(__file__)
dirname = os.path.dirname(__file__)

file_path_category = os.path.join(dirname, 'category_id_clean.txt')
with open(file_path_category, 'r') as file:
    class_labels = [line.strip().replace("\t", "") for line in file]

excel_file_path = 'model/Food_Dataset.xlsx'
df = pd.read_excel(excel_file_path, sheet_name='Carbon Emission')

"""
download the model from this link 
https://drive.google.com/uc?id=1Hi5ND78_yyD5dQlVN6WlcpDMw6TYwACG
and save it to model/model_segmentation.h5
"""

filename = os.path.join(dirname, 'model_segmentation.h5')
model = tf.keras.models.load_model(filename, custom_objects={'KerasLayer':hub.KerasLayer})

# Preprocess the Image
def preprocess_image(img):
  img = img.resize((224, 224))
  img = img.convert('RGB') 
  img_arr = tf.keras.preprocessing.image.img_to_array(img)
  img_arr = np.expand_dims(img_arr, axis=0)
  img_arr = img_arr / 255.0
  return img_arr

# Predicting the label
def predict_image_sgmnt(img):
  img_arr = preprocess_image(img)
  prediction = model.predict(img_arr)
  predict_mask = np.argmax(prediction, axis=3)
  unique_values, counts = np.unique(predict_mask, return_counts=True)
  counts_dict = dict(zip(unique_values, counts))
  sorted_counts = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
  ingredient = []

  for value, count in sorted_counts:
    if (count/(224*224))*100 >=3:
      ingredient.append(class_labels[value])

  class_to_delete = ["background", "sauce", "juice", "ice cream", "other ingredients"]
  for i in class_to_delete:
    if i in ingredient:
      ingredient.remove(i)

  return ingredient