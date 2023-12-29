import cv2
from keras.models import load_model
import numpy as np
import random

cpu_choice = random.choice(["rock", "paper", "scissors"])
player_choice = ""

model = load_model("rps_model.h5")

cam = cv2.VideoCapture(0)

cv2.namedWindow("Rock Paper Scissors")

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
            player_choice = "paper"
        elif index == 1:
            player_choice = "rock"
        elif index == 2:
            player_choice = "scissors"
        
        print("ESC to proceed")
        print("SPACE to change the decision")
        
        # img_name = "opencv_frame_{}.png".format(img_counter)
        # cv2.imwrite(img_name, frame)
        # print("{} written!".format(img_name))
        # img_counter += 1

cam.release()

cv2.destroyAllWindows()

print("="*10)
print(prediction)
print("="*10)
print()
print(f"Player: {player_choice}\nComputer: {cpu_choice}")
if player_choice == cpu_choice:
    print("Tie")
else:
    if player_choice == "rock":
        if cpu_choice == "paper":
            print("Computer wins")
        elif cpu_choice == "scissors":
            print("Player wins")
    elif player_choice == "paper":
        if cpu_choice == "scissors":
            print("Computer wins")
        elif cpu_choice == "rock":
            print("Player wins")
    elif player_choice == "scissors":
        if cpu_choice == "rock":
            print("Computer wins")
        elif cpu_choice == "paper":
            print("Player wins!")
