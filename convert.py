#code to convert tf model into tf-lite
import tensorflow as tf
from tensorflow.lite.python.convert import convert_saved_model as _convert_saved_model
from tensorflow.keras.models import load_model

model = tf.keras.models.load_model('0_5model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open('model.tflite', 'wb').write(tflite_model)