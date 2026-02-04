from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_with_image(img_path, pre_trained_model, input_shape=(64, 64)):
    model_input_shape = pre_trained_model.input_shape
    print(f"Model input shape: {model_input_shape}")

    if input_shape is None:
        input_shape = model_input_shape[1:3]
    else:
        input_shape = (input_shape[0], input_shape[1])
    print(f"Input image shape: {input_shape}")

    if len(model_input_shape) > 3:
        channels = model_input_shape[3]
    else:
        channels = 1

    # load image with correct channels
    color_mode = 'grayscale' if channels == 1 else 'rgb'
    
    img = image.load_img(img_path, target_size=input_shape, color_mode=color_mode)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
    x2 = x 
    print('Input image shape:', x.shape)
    imshow(img)
    prediction = pre_trained_model.predict(x2)
    num_of_classes = pre_trained_model.output_shape[1]
    class_labels = [f"p({i})" for i in range(num_of_classes)]
    print("Class:", np.argmax(prediction))
    print(f"Class prediction vector [{','.join(class_labels)}] = {prediction}")
