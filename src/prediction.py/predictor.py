import cv2
from src.face_recognition import faceRecognitionPipeline
import matplotlib.pyplot as plt

### Face recognition prediction
pred_img, pred_dict = faceRecognitionPipeline('getty_test.jpg')

img_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

#generate the report
for i in range(len(pred_dict)):
    obj_gray = pred_dict[i]['roi'] #gray scale-- crop image
    obj_eig = pred_dict[i]['eig_img'].reshape(100,100)
    plt.subplot(1,2,1)
    plt.imshow(obj_gray, cmap='gray')
    plt.title('Gray ScaleImage')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(obj_eig, cmap='gray')
    plt.title('Eigen Image')
    plt.axis('off')
    
    plt.show()
    print("Predicted Gender = ", pred_dict[i]['prediction_name'])
    print("Predicted score ={:,.2f}% ".format(pred_dict[i]['score']*100))
    print("-"*100)
    
### Real Time Gender Recognition    
cap = cv2.VideoCapture("video.mp4") #0--> access webcamera, 1--> external camera, 2 -->path of img,video

while True:
    ret, frame = cap.read()
    
    if ret == False:
        break
        
    pred_img, pred_dict = faceRecognitionPipeline(frame, path=False)

    cv2.imshow("prediction", pred_img)
    if cv2.waitKey(1)== ord('q'):
           break 

cv2.release()
cv2.destroyAllWindows()