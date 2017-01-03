import cv2
import numpy as np


def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) <= 20.:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


def get_square(xs, ys):
    res = 0.
    x_arr = []
    y_arr = []
    for _ in zip(xs, xs[1:] + (xs[0],)):
        x_arr.append(_[0] + _[1])

    for _ in zip(ys, ys[1:] + (ys[0],)):
        y_arr.append(_[0] - _[1])

    for a, b in zip(x_arr, y_arr):
        res += a * b
    return 0.5 * abs(res)


def smooth_path(rat_path, avg_step=7):
    for _ in range(len(rat_path)):
        for idx, (p1, p2) in enumerate(zip(rat_path, rat_path[1:])):
            if ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5 < 3:
                rat_path.pop(idx)
                break

    avg_pth = []
    for idx in range(avg_step, len(rat_path), avg_step):
        avg_x = sum([p1 for (p1, _) in rat_path[idx - avg_step: idx]]) / avg_step
        avg_y = sum([p2 for (_, p2) in rat_path[idx - avg_step: idx]]) / avg_step

        avg_pth.append((avg_x, avg_y))

    return avg_pth


def main():
    # noinspection PyArgumentList
    cap = cv2.VideoCapture("video/00023.mp4")
    out_curr = cv2.VideoWriter("solution_final.avi", cv2.VideoWriter_fourcc(*'PIM1'), 25, (1280, 720))
    _, back = cap.read()

    gauss_size = (19, 19)
    back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    back = cv2.GaussianBlur(back, gauss_size, 0)

    rat_path = []
    fr_counter = 0
    mc_x, mc_y, mc_r = None, None, None
    mc_mask = np.zeros((back.shape[0], back.shape[1]), dtype=back.dtype) + 255

    while cap.isOpened():
        _, frame = cap.read()
        fr_counter += 1
        orig = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if mc_r is None:
            circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 2., 250, minRadius=250, maxRadius=300)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    mc_x, mc_y, mc_r = (x, y, r + 5)

                    for i in range(mc_mask.shape[0]):
                        for j in range(mc_mask.shape[1]):
                            if ((mc_y - i) ** 2 + (mc_x - j) ** 2) ** 0.5 > mc_r:
                                mc_mask[i][j] = 0
                    break

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

        thresh = cv2.dilate(thresh, None, iterations=5)
        thresh &= mc_mask

        thresh = cv2.Canny(thresh, 30, 200)
        img_cnt, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.circle(orig, (mc_x, mc_y), mc_r, (0, 0, 200), 4)

        if len(contours) > 1:
            clust = np.zeros((len(contours), 1))
            for i, cnt1 in enumerate(contours):
                x = i
                if i != len(contours) - 1:
                    for j, cnt2 in enumerate(contours[i + 1:]):
                        x += 1
                        if find_if_close(cnt1, cnt2):
                            val = min(clust[i], clust[x])
                            clust[x] = clust[i] = val
                        else:
                            if clust[x] == clust[i]:
                                clust[x] = i + 1

            unified = []
            maximum = int(clust.max()) + 1
            for i in xrange(maximum):
                pos = np.where(clust == i)[0]
                if pos.size != 0:
                    cont = np.vstack(contours[i] for i in pos)
                    hull = cv2.convexHull(cont)
                    unified.append(hull)

            all_cnts = []
            for h in unified:
                xs, ys = zip(*[(arr[0], arr[1])for arr, in h])
                all_cnts.append((h, get_square(xs, ys)))

            all_cnts = sorted(all_cnts, key=lambda x: x[1], reverse=True)[0][0]
            cv2.drawContours(orig, [all_cnts], 0, (255, 0, 0), 2)

            mm = cv2.moments(all_cnts)
            if abs(mm["m00"]) < 10e-5:
                continue
            c_x = int(mm["m10"] / mm["m00"])
            c_y = int(mm["m01"] / mm["m00"])

            cv2.circle(orig, (c_x, c_y), 3, (0, 0, 255), -1)
            rat_path.append((c_x, c_y))

            avg_pth = smooth_path(rat_path)
            dst = 0.0
            for (p1, p2) in zip(avg_pth, avg_pth[1:]):
                cv2.line(orig, p1, p2, (0, 255, 255), thickness=2)
                dst += ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

            cv2.putText(orig, 'Path: {}'.format(int(dst)), (20, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("contour", orig)
        out_curr.write(orig)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_curr.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    exit(main())
