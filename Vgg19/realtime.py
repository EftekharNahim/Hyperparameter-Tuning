import tensorflow as tf
import numpy as np
import cv2
from keras.utils import load_img, img_to_array
from keras.models import model_from_json
import os


# load json and create model
json_file = open('DenseNet2_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("DenseNet2_model.h5")


def getClassName(classNo):
    if   classNo == 0: return 'Glass'
    elif classNo == 1: return 'Metal'
    elif classNo == 2: return 'Paper'
    elif classNo == 3: return 'Plastic'
    elif classNo == 4: return 'Trash'




threshold = 75 
font = cv2.FONT_HERSHEY_SIMPLEX


# VIDEOS_DIR = os.path.join('../../YOLO/YOLOV8/', 'videos')
# video_path = os.path.join(VIDEOS_DIR, 'Glass.mp4')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # READ IMAGE
    success, imgOrignal = cap.read()

    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    # # PREDICT IMAGE
    # img_array = tf.keras.preprocessing.image.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0) 
    # predictions = model.predict(img_array)


    img=img_to_array(imgOrignal)
    img = tf.keras.preprocessing.image.smart_resize(img, (224, 224))
    img = img / 255.0
    img =np.expand_dims(img, axis =0)
    predictions = model.predict(img)
    print(predictions)


    probabilityValue = predictions[0][np.argmax(predictions)]*100
    classIndex = np.argmax(predictions)
    # classType = classes[np.argmax(predictions)]
    classType = getClassName(np.argmax(predictions))

    # print("Prediction: ", classes[np.argmax(predictions)], f"{predictions[0][np.argmax(predictions)]*100}%")

    if probabilityValue > threshold:
        print(classType)
        cv2.putText(imgOrignal,str(classIndex)+" "+str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(probabilityValue)+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()