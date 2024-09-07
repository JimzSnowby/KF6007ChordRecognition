import cv2
import os
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands


image_x, image_y = 500, 500
output_dir ='src/dataset/Gestures'


def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main(c_id):
    total_pics = 1200
    hands = mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    create_folder(output_dir + '/' + str(c_id))
    pic_no = 0
    flag_start_capturing = False


    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(image_rgb)

        # Take photos when hands detected, then crop to hand only
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                h, w, _ = image.shape
                bbox = min([lm.x for lm in hand_landmarks.landmark]), \
                        max([lm.x for lm in hand_landmarks.landmark]), \
                        min([lm.y for lm in hand_landmarks.landmark]), \
                        max([lm.y for lm in hand_landmarks.landmark])
                bbox = np.array([bbox[0] * w, bbox[1] * w, bbox[2] * h, bbox[3] * h]).astype(int)
                padding = 20
                bbox = np.array([max(bbox[0] - padding, 0),
                                min(bbox[1] + padding, w),
                                max(bbox[2] - padding, 0),
                                min(bbox[3] + padding, h)])
                
                hand_img = image[bbox[2]:bbox[3], bbox[0]:bbox[1]]
                
            pic_no += 1
            cv2.imwrite(output_dir + '/' + str(c_id) + '/' + str(pic_no) + '.jpg', hand_img)
            cv2.putText(image, str(pic_no), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            flag_start_capturing = not flag_start_capturing
            frames= 0 if not flag_start_capturing else frames


        if flag_start_capturing:
            frames += 1


        if pic_no == total_pics:
            break


        cv2.imshow('Capturing gesture', image)
    cap.release()
    cv2.destroyAllWindows()


c_id = input('Enter Chord: ')
main(c_id)