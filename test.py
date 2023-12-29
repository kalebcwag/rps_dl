import cv2
from keras.models import load_model
import numpy as np

model = load_model("rps_model.h5")

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), 2)
    
    cv2.imshow("test", frame)

    key = cv2.waitKey(1)
    if key%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif key%256 == 32:
        # SPACE pressed
        prediction_image = frame[100:300, 100:300]
        
        prediction_image = cv2.resize(prediction_image, (150, 150))
        
        prediction_image = np.array(prediction_image)
        prediction_image = np.expand_dims(prediction_image, axis=0)
        prediction_image = prediction_image.astype("float32")/255.0
        
        prediction = model.predict(prediction_image)[0]
        
        index = np.argmax(prediction)
        if index == 0:
            print("It's a picture of paper")
        elif index == 1:
            print("It's a picture of rock")
        elif index == 2:
            print("It's a picture of scissors")
        
        # img_name = "opencv_frame_{}.png".format(img_counter)
        # cv2.imwrite(img_name, frame)
        # print("{} written!".format(img_name))
        # img_counter += 1

cam.release()

cv2.destroyAllWindows()