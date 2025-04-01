#!/usr/bin/env python3
import cv2
import numpy as np

def nothing(x):
    pass

def main():
    # 기본 카메라 (0번 장치) 사용. 필요시 다른 장치 번호나 비디오 파일로 변경.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
    
    # Lower HSV 트랙바 (H: 0~179, S: 0~255, V: 0~255)
    cv2.createTrackbar("Lower H", "HSV Tuner", 0, 179, nothing)
    cv2.createTrackbar("Lower S", "HSV Tuner", 0, 255, nothing)
    cv2.createTrackbar("Lower V", "HSV Tuner", 0, 255, nothing)
    
    # Upper HSV 트랙바
    cv2.createTrackbar("Upper H", "HSV Tuner", 179, 179, nothing)
    cv2.createTrackbar("Upper S", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("Upper V", "HSV Tuner", 255, 255, nothing)
    
    print("ESC를 누르면 종료됩니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽어올 수 없습니다.")
            break

        # 원본 영상과 HSV 영상 생성
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 트랙바에서 현재 HSV 값 읽기
        lower_h = cv2.getTrackbarPos("Lower H", "HSV Tuner")
        lower_s = cv2.getTrackbarPos("Lower S", "HSV Tuner")
        lower_v = cv2.getTrackbarPos("Lower V", "HSV Tuner")
        upper_h = cv2.getTrackbarPos("Upper H", "HSV Tuner")
        upper_s = cv2.getTrackbarPos("Upper S", "HSV Tuner")
        upper_v = cv2.getTrackbarPos("Upper V", "HSV Tuner")
        
        lower_hsv = np.array([lower_h, lower_s, lower_v])
        upper_hsv = np.array([upper_h, upper_s, upper_v])
        
        # 마스크 생성 및 결과 영상 계산
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # 결과 영상 창 표시
        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

