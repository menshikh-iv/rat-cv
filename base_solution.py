import cv2
import numpy as np


def main():
    # noinspection PyArgumentList
    cap = cv2.VideoCapture("video/00023.avi")
    out = cv2.VideoWriter("main_stages_2.avi", cv2.VideoWriter_fourcc(*'PIM1'), 25,
                          (640 * 2, 480 * 2), isColor=False)
    _, back = cap.read()

    gauss_size = (19, 19)
    back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    back = cv2.GaussianBlur(back, gauss_size, 0)

    rat_path = []
    fr_counter = 0
    while cap.isOpened():
        _, frame = cap.read()
        fr_counter += 1
        orig = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, gauss_size, 0)

        _output_gray_blur = frame.copy()
        cv2.putText(_output_gray_blur, 'Gray + Blur (19, 19)', (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        delta = cv2.absdiff(back, frame)
        _output_diff = delta.copy()
        cv2.putText(_output_diff, 'Diff', (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        _, thresh = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)
        _output_thresh = thresh.copy()
        cv2.putText(_output_thresh, 'Thresh binary (20, 255)', (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        thresh = cv2.dilate(thresh, None, iterations=10)
        _output_dilate = thresh.copy()
        cv2.putText(_output_dilate, 'Dilate', (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        img_cnt, contours, _ = cv2.findContours(cv2.Canny(thresh, 30, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # TODO select best contour (largest not works)
        # largest_cnt = sorted(contours, key=cv2.contourArea, reverse=True)

        cnt_idx = 0
        for idx, c in enumerate(contours):
            if idx == cnt_idx:
                mm = cv2.moments(c)
                if abs(mm["m00"]) < 10e-5:
                    continue
                c_x = int(mm["m10"] / mm["m00"])
                c_y = int(mm["m01"] / mm["m00"])
                cv2.circle(orig, (c_x, c_y), 3, (0, 0, 255), -1)
                rat_path.append((c_x, c_y))

        avg_pth = []
        avg_step = 15
        for idx in range(avg_step, len(rat_path), avg_step):
            avg_x = sum([p1 for (p1, _) in rat_path[idx - avg_step: idx]]) / avg_step
            avg_y = sum([p2 for (_, p2) in rat_path[idx - avg_step: idx]]) / avg_step

            avg_pth.append((avg_x, avg_y))

        dst = 0.0
        for (p1, p2) in zip(avg_pth, avg_pth[1:]):
            cv2.line(orig, p1, p2, (0, 255, 255), thickness=2)
            dst += ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

        cv2.putText(orig, 'Path: {}'.format(int(dst)), (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.drawContours(orig, contours, cnt_idx, (0, 240, 0), thickness=3)

        """
        circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=260, maxRadius=300)

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(orig, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(orig, (i[0], i[1]), 2, (0, 0, 255), 3)
        """


        cv2.imshow("contour", orig)
        cv2.imshow('frame', thresh)

        """
        h1 = np.hstack((_output_gray_blur, _output_diff))
        h2 = np.hstack((_output_thresh, _output_dilate))
        full = np.vstack((h1, h2))

        cv2.imshow("FULL", full)
        out.write(full)
        """

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    exit(main())
