import cv2
import numpy as np

frameWidth = 640
frameHeight = 480

# capturing Video from Webcam
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# set brightness, id is 10 and
# value can be changed accordingly
cap.set(10,150)

def resize(img):
    dim = None
    (h, w) = img.shape[:2]
    r = frameHeight / float(h)
    dim = (int(w * r), frameHeight)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

if __name__ == "__main__":

    while True:
        success, img = cap.read()
        imgResult = img.copy()
        imgResult = cv2.flip(imgResult, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(gray, (3,3),0)

        #Roberts Filter
        kernelx = np.array([[-1,0],[0,1]])
        kernely = np.array([[0,-1],[1,0]])

        img_robertsx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_robertsy = cv2.filter2D(img_gaussian, -1 , kernely)

        #Prewitt Filter
        kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
        
        img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        
        #Sobel Filters
        kernelx= np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        kernely=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

        img_sobelx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_sobely = cv2.filter2D(img_gaussian, -1, kernely)

        cv2.imshow("Roberts", resize(5*(img_robertsx + img_robertsy)))
        cv2.imshow("Prewitt", resize(img_prewittx + img_prewitty))
        cv2.imshow("Sobel", resize(img_sobelx + img_sobely))

        key = cv2.waitKey(10)
        if key == ord('q'):
            break