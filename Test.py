import cv2
import MobileNetSSDModule as mnssdm
import socket 
#import asyncio
#from apscheduler.schedulers.background import BackgroundScheduler
import time



def take_ss():
    #create webcam
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)

    model = mnssdm.MobileNetSSD("ssd-mobilenet-v2", threshold=0.5)

    #sched = BackgroundScheduler()
    timeout = time.time() + 8
    timeout2 = time.time() + 9
    while True:
        success, img = cap.read()

        if success:
                    

            objects, img = model.detect(img, display=True)

                
            #print(objects)
            #convert the image tagged to a numpy image
            #img = jetson.utils.cudaToNumpy(imgCuda)
            

            #cv2.imshow("Video", img)

            if time.time() > timeout:
                save_image(img) 

            if time.time() > timeout2:
                break;           
            
            if cv2.waitKey(1) >= 0:
                break

        else:
            print("hola")
            break

    cap.release()
    cv2.destroyAllWindows()

def save_image(imagen):
    cv2.imwrite("imagenes/imagen.jpg", imagen)

def open_socket():

    print("Socket iniciado")

    hostname = socket.gethostbyname("0.0.0.0")

    mi_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mi_socket.bind((hostname,8001))
    mi_socket.listen(1)

    while True:
        
        conexion, addr = mi_socket.accept()

        print("Nueva conexion")
        print(addr)

        peticion = conexion.recv(1024)

        print(peticion.decode())

        if peticion:
            take_ss()
            time.sleep(2.0)

            imagen = open("imagenes/imagen.jpg", "rb")

            bytes = imagen.read()

            mi_socket.sendall(bytes)

        print("terminada la foto")

        #conexion.send("Hola, estamos conectados".encode())

        conexion.close()


if __name__ == "__main__":
    open_socket()

