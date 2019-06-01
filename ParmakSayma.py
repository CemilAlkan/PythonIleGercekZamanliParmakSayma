import math
import cv2
import numpy as np

cam = cv2.VideoCapture(0)
cam.set(10 , 300)

def ParmakSay(noktalar, ciz):
    hull = cv2.convexHull(noktalar, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(noktalar, hull)
        if defects is not None:
            sayac = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i][0]
                bas = tuple(noktalar[s][0])
                son = tuple(noktalar[e][0])
                uzaklık = tuple(noktalar[f][0])
                a = math.sqrt((son[0] - bas[0]) ** 2 + (son[1] - bas[1]) ** 2)
                b = math.sqrt((uzaklık[0] - bas[0]) ** 2 + (uzaklık[1] - bas[1]) ** 2)
                c = math.sqrt((son[0] - uzaklık[0]) ** 2 + (son[1] - uzaklık[1]) ** 2)
                aci = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                if aci <= math.pi / 2:
                    sayac += 1
                    cv2.circle(ciz, uzaklık, 8, [211, 84, 0], -1)
            if sayac > 0:
                return True, sayac+1
            else:
                return True, 1
    return False, 0

def openCam():
    while cam.isOpened():
        fr,goruntu = cam.read()

        #üzerinde işlem yapacagımız alnan belirliyoruz
        kare = goruntu[100:400 , 100:400]
        kesilmis_kare = goruntu[150:350, 150:350]

        #RGB görüntümüzü HSV moduna dönüştürüyoruz
        hsv = cv2.cvtColor(kesilmis_kare, cv2.COLOR_BGR2HSV)

        kernal = np.ones((3,3), np.uint8)
        min_deri = np.array([0, 20, 50], dtype=np.uint8)
        max_deri = np.array([15, 255, 255], dtype=np.uint8)

        #Sadece seçilen lower ve upper değerler arasındaki renklerin algılanmasını sağlıyoruz
        renk_filtresi = cv2.inRange(hsv, min_deri, max_deri)

        #Nesnelerin üzerinde oluşan boşlukların kapatılması için kullandık
        renk_filtresi = cv2.morphologyEx(renk_filtresi, cv2.MORPH_CLOSE, kernal)

        #Görüntüdeki beyaz bölgeyi veya ön plandaki nesnenin boyutunu artırır.
        renk_filtresi = cv2.dilate(renk_filtresi, kernal, iterations=5)

        #Görüntünün contoursları belirlenir
        blank,contours, blank1 = cv2.findContours(renk_filtresi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxalan = -1
        if length > 0:
            for i in range(length):
                temp = contours[i]
                alan = cv2.contourArea(temp)
                if alan > maxalan:
                    maxalan = alan
                    ci = i

            noktalar = contours[ci]
            hull = cv2.convexHull(noktalar)
            #kesilmiş karenin boyutunda 0 lardan oluşan bir matris oluşturulur
            ciz = np.zeros(kesilmis_kare.shape, np.uint8)
            # En büyük alana sahip şeklin contoursları çizdirilir
            cv2.drawContours(ciz, [noktalar], 0, (0, 255, 0), 2)
            cv2.drawContours(ciz, [hull], 0, (0, 0, 255), 3)

            bos, cnt = ParmakSay(noktalar, ciz)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(ciz, str(cnt), (25, 25), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('cıkıs', ciz)






        cv2.imshow("baslık", renk_filtresi)
        cv2.imshow("kare", goruntu)

        if cv2.waitKey(1)&0xFF == ord("q"):
            break
openCam()
cam.release()
cv2.destroyAllWindows()