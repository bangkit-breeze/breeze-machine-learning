import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

file_path = 'model/category_id_clean.txt' 
with open(file_path, 'r') as file:
    class_labels = [line.strip().replace("\t", "") for line in file]

"""
download the model from this link 
https://drive.google.com/uc?id=1Hi5ND78_yyD5dQlVN6WlcpDMw6TYwACG
and save it to model/Segmentation_model.h5
"""

model = tf.keras.models.load_model("model/Segmentation_model.h5", custom_objects={'KerasLayer':hub.KerasLayer})

# Preprocess the Image
def preprocess_image(img):
  img = img.resize((224, 224))
  img = img.convert('RGB') 
  img_arr = tf.keras.preprocessing.image.img_to_array(img)
  img_arr = np.expand_dims(img_arr, axis=0)
  img_arr = img_arr / 255.0
  return img_arr

# Predicting the label
def predict_image_segmentation(img):
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
  return ingredient