import jetson_inference
import jetson_utils
import cv2
import numpy as np

class MobileNetSSD():
    def __init__(self,path, threshold=0.5):
        self.path = path
        self.threshold = threshold
        #creating the network
        self.net = jetson_inference.detectNet(self.path, self.threshold)

    def detect(self, img, display=True):
        #convert the image to imageCuda format
        img = np.array(img)
        imgCuda = jetson_utils.cudaFromNumpy(img)
        detections = self.net.Detect(imgCuda, overlay="OVERLAY_NONE")

        objects = []

        for d in detections:

            #class name
            className = self.net.GetClassDesc(d.ClassID)
            objects.append([className,d])

            if display == True:
                #coordinates
                x1,y1,x2,y2 = int(d.Left),int(d.Top),int(d.Right),int(d.Bottom)
                #center values
                cy,cx = int(d.Center[0]), int(d.Center[1])

                cv2.rectangle(img,(x1,y1),(x2,y2), (255,0,255),2)
                """
                cv2.circle(img, (cx,cy), 5, (255,0,255),cv2.FILLED)
                cv2.line(img,(x1,cy), (x2,cy), (255,0,255),1)
                cv2.line(img,(cx,y1), (cx,y2), (255,0,255),1)"""

                cv2.putText(img, className, (x1+5,y1+15), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,0,255),2)
                #cv2.putText(img,f"FPS: " + int(self.net.GetNetworkFPS()), (30,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),2)

        return objects, img

def main():
    
    #create webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("camera not opened")
        exit()
    cap.set(3,640)
    cap.set(4,480)

    model = MobileNetSSD("ssd-mobilenet-v2", threshold=0.5)

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


if __name__ == "__main__":
    main()

    


