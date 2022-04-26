import cv2
import mediapipe as mp
import math

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(min_detection_confidence=0.8) as face_detection:   #model_selection=0,
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                print(detection)
                bound_area = detection.location_data.relative_bounding_box
                right_eyes = detection.location_data.relative_keypoints[0]  #右眼睛
                left_eyes = detection.location_data.relative_keypoints[1]   #

                nose = detection.location_data.relative_keypoints[2]
                final_xmin = bound_area.xmin + 1/4 * bound_area.width
                final_xmax = bound_area.xmin + 3/4 * bound_area.width
                if final_xmin <= nose.x <= final_xmax:
                    print('正脸')
                else:
                    print('侧脸')

                # l_n = math.sqrt((left_eyes.x - nose.x)**2 + (left_eyes.y - nose.y)**2)
                # r_n = math.sqrt((right_eyes.x - nose.x) ** 2 + (right_eyes.y - nose.y) ** 2)
                # print(l_n,r_n)

                mp_drawing.draw_detection(image, detection)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()