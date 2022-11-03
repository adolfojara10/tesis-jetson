import jetson.inference
import jetson.utils
import cv2

#creating the network
net = jetson.inference.detectNet("ssd-mobilenet-v2",threshold=0.5)

#create webcam
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    success, img = cap.read()

    #convert the image to imageCuda format
    imgCuda = jetson.utils.cudaFromNumpy(img)

    detections = net.Detect(imgCuda)

    #convert the image tagged to a numpy image
    #img = jetson.utils.cudaToNumpy(imgCuda)

    for d in detections:
        #coordinates
        x1,y1,x2,y2 = int(d.Left),int(d.Top),int(d.Right),int(d.Bottom)

        #class name
        className = net.GetClassDescription(d.ClassID)

        cv2.rectangle(img,(x1,y1),(x2,y2), (255,0,255),2)
        cv2.putText(img, className, (x1+5,y1+15), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,0,255),2)


    cv2.imshow("Video", img)
    cv2.waitKey(1)


