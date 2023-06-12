import numpy as np
import cv2


def add_noise():
    noise_multiplier = int(input('Введите коэфициент для добавляемого шума (целое число 1-99): '))
    if noise_multiplier > 99:
        noise_multiplier = 0.99
    elif noise_multiplier < 1:
        noise_multiplier = 0.01
    else:
        noise_multiplier /= 100

    img = cv2.imread(r'variant-5.jpg', cv2.IMREAD_COLOR)
    image_arr = np.array(img/255, dtype=float)
    noise_arr = np.random.normal(0, noise_multiplier, img.shape)

    image_final = image_arr + noise_arr

    cv2.imshow(f'Noise {noise_multiplier}', image_final)
    cv2.waitKey(0)


def circle_tracking():

    video = cv2.VideoCapture(r'sample.mp4')
    detector = cv2.createBackgroundSubtractorMOG2(history=100)

    while True:
        ret, frame = video.read()
        mask = detector.apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        object_contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for countur in object_contour:
            area = cv2.contourArea(countur)
            if area > 450:
                (x, y), r = cv2.minEnclosingCircle(countur)
                center = (int(x), int(y))
                r = int(r)

                # Если метка попадает в левый верхний угол (область 50 на 50 очень мала,
                # поэтому для демонстрации я использую другие параметры), то она окрашивается в синий
                # Если попадает в правый нижний, то в красный

                if center[0] < 250 and center[1] < 210:
                    cv2.circle(frame, center, r, (255, 0, 0), 3)
                elif center[0] > 340 and center[1] > 300:
                    cv2.circle(frame, center, r, (0, 0, 255), 3)
                else:
                    cv2.circle(frame, center, r, (0, 0, 0), 3)

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(30)
        if key == 27:
            break

    video.release()
    cv2.destroyAllWindows()


add_noise()
circle_tracking()
