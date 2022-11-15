import cv2
import MobileNetSSDModule as mnssdm



#create webcam
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

model = mnssdm.MobileNetSSD("ssd-mobilenet-v2", threshold=0.5)

while True:
    success, img = cap.read()

    if success:
                

        objects, img = model.detect(img, display=True)

            
        print(objects)
        #convert the image tagged to a numpy image
        #img = jetson.utils.cudaToNumpy(imgCuda)
        

        cv2.imshow("Video", img)
        if cv2.waitKey(1) >= 0:
            break

    else:
        print("hola")
        break

cap.release()
cv2.destroyAllWindows()