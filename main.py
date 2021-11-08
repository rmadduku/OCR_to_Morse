import cv2
import pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def reading(image):
    img = image
    resize_x =1
    resize_y =1
    cv2.waitKey()


    # # sharpen image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # resize image to 3 times as large as original for better readability
    gray = cv2.resize(gray, None, fx=resize_x, fy=resize_y, interpolation=cv2.INTER_CUBIC)
    gray = cv2.convertScaleAbs(gray, alpha=3, beta=100)
    cv2.imshow("Gray", gray)
    cv2.waitKey(0)

    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    cv2.imshow("thresh",thresh)
    cv2.waitKey(0)

    # morpological open/closing
    kernel = np.ones((3,3),np.uint8)
    test_img = cv2.dilate(thresh,kernel,iterations = 2)
    test_img = cv2.erode(test_img,kernel,iterations = 1)
    # test_img = cv2.morphologyEx(test_img,cv2.MORPH_OPEN,kernel)
    # test_img = cv2.morphologyEx(test_img,cv2.MORPH_CLOSE,kernel)
    cv2.imshow("closing", test_img)
    cv2.waitKey(0)



    ## Detecting numbers

    height, width,_ = img.shape
    # cong = r'--oem 3 --psm 6 outputbase digits'
    boxes = pytesseract.image_to_data(test_img)
    img = cv2.resize(img, None, fx=resize_x, fy=resize_y, interpolation=cv2.INTER_CUBIC)
    words = []
    for z,b in enumerate(boxes.splitlines()):
        if(z>0):
            b = b.split()
            if len(b) == 12:
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(img,(x,y),(w+x,h+y),[255,0,0],1)
                cv2.putText(img,b[11],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,[255,0,50],2)
                words.append(b[11])

    cv2.imshow("res",img)
    cv2.waitKey()
    return words

MORSE_CODE_DICT = {'A': '.-', 'B': '-...',
                   'C': '-.-.', 'D': '-..', 'E': '.',
                   'F': '..-.', 'G': '--.', 'H': '....',
                   'I': '..', 'J': '.---', 'K': '-.-',
                   'L': '.-..', 'M': '--', 'N': '-.',
                   'O': '---', 'P': '.--.', 'Q': '--.-',
                   'R': '.-.', 'S': '...', 'T': '-',
                   'U': '..-', 'V': '...-', 'W': '.--',
                   'X': '-..-', 'Y': '-.--', 'Z': '--..',
                   '1': '.----', '2': '..---', '3': '...--',
                   '4': '....-', '5': '.....', '6': '-....',
                   '7': '--...', '8': '---..', '9': '----.',
                   '0': '-----', ', ': '--..--', '.': '.-.-.-',
                   '?': '..--..', '/': '-..-.', '-': '-....-',
                   '(': '-.--.', ')': '-.--.-'}

def encrypt(message):
    cipher = ''
    for letter in message:
        if letter != ' ':
            if not letter in MORSE_CODE_DICT:
                cipher+="N\A"
            else:
                cipher += MORSE_CODE_DICT[letter] + ' '
        else:
            cipher += ' '

    return cipher



def main():
    words = reading(cv2.imread('testnumbers.PNG'))

    for word in words:
        result = encrypt(word.upper())
        print(result)

# Executes the main function
if __name__ == '__main__':
    main()