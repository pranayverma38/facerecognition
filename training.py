import numpy as numpy
import cv2
import os

import face_recognition as fr 


test_img=cv2.imread(r"D:\project\face recognition\test_img.jpeg")

faces_detected,gray_img=fr.faceDetection(test_img)
print("Face Detected: ",faces_detected)



faces,faceID=fr.labels_for_training_data(r"D:\project\face recognition\images")
face_recognizer=fr.train_Classifier(faces,faceID)
face_recognizer.save(r"D:\project\face recognition\trainingData.yml")

name={0:'pranay'}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+w,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print(label)
    print(confidence)
    fr.draw_rect(test_img,face)
    predict_name=name[label]
    fr.put_text(test_img,predict_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow("face detection",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

