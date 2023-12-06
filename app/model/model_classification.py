import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

class_labels = [
   'anggur', 'apel', 'ayam goreng', 'bakso', 'bubur-ayam', 'bubur-biji-salak', 'donuts', 'dumplings', 'hamburger', 'ikan-bakar', 'ikan-goreng', 'kentang goreng', 'kopi', 'martabak-manis', 'mie ayam', 'nasi-goreng', 'pisang', 'pisang goreng', 'rawon', 'rendang', 'roti', 'sate ayam', 'soto', 'steak', 'susu', 'tahu goreng', 'teh', 'telur', 'tempe-goreng', 'udang goreng'
]

# Load model
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'model_classification.h5')
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
def predict_image_clf(img):
  img_arr = preprocess_image(img)
  predictions = model.predict(img_arr)
  predicted_class_index = np.argmax(predictions)
  predicted_class_label = class_labels[predicted_class_index]
  confidence = np.max(predictions)
  return predicted_class_label, confidence