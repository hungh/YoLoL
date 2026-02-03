from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_with_image(img_path, pre_trained_model, input_shape=(64, 64)):
    img = image.load_img(img_path, target_size=input_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
    x2 = x 
    print('Input image shape:', x.shape)
    imshow(img)
    prediction = pre_trained_model.predict(x2)
    print("Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ", prediction)
    print("Class:", np.argmax(prediction))
