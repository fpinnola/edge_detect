import cv2
import numpy as np
from numpy.fft import fft2, ifft2

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

def fft_convolve2d(x,y):
    """ 2D convolution, using FFT"""
    fr = fft2(x)
    fr2 = fft2(np.flipud(np.fliplr(y)))
    m,n = fr.shape
    cc = np.real(ifft2(fr*fr2))
    cc = np.roll(cc, int(-m/2+1),axis=0)
    cc = np.roll(cc, int(-n/2+1),axis=1)
    return cc

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

        #Custom Sobel using numpy
        input_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        padding_top = int((input_y - 3) / 2)
        padding_bottom = input_y - padding_top - 3

        padding_left = int((input_x - 3)/2)
        padding_right = input_x - padding_left - 3

        filterx = np.pad(kernelx, ((padding_top, padding_bottom), (padding_left, padding_right)), 'constant')

        filtery = np.pad(kernely, ((padding_top, padding_bottom), (padding_left, padding_right)), 'constant')
        img_sobelx_np = fft_convolve2d(img_gaussian, filterx)
        img_sobely_np = fft_convolve2d(img_gaussian, filtery)


        cv2.imshow("Roberts", resize(5*(img_robertsx + img_robertsy)))
        cv2.imshow("Prewitt", resize(img_prewittx + img_prewitty))
        cv2.imshow("Sobel", resize(img_sobelx + img_sobely))
        cv2.imshow("Sobel2", resize(img_sobelx_np + img_sobely_np))

        key = cv2.waitKey(10)
        if key == ord('q'):
            break