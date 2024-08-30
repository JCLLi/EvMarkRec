import math
import random

import cv2
import numpy as np


def imgZRotate(img, x, y, z):
    proj2dto3d = np.array([[1, 0, -img.shape[1] / 2],
                           [0, 1, -img.shape[0] / 2],
                           [0, 0, 0],
                           [0, 0, 1]], np.float32)

    rx = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)  # 0
    ry = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)  # 0

    trans = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 260],  # 400 to move the image in z axis
                      [0, 0, 0, 1]], np.float32)

    proj3dto2d = np.array([[200, 0, img.shape[1] / 2, 0],
                           [0, 200, img.shape[0] / 2, 0],
                           [0, 0, 1, 0]], np.float32)

    ax = float(x * (math.pi / 180.0))  # 0
    ay = float(y * (math.pi / 180.0))
    az = float(z * (math.pi / 180.0))  # 0

    rx[1, 1] = math.cos(ax)  # 0
    rx[1, 2] = -math.sin(ax)  # 0
    rx[2, 1] = math.sin(ax)  # 0
    rx[2, 2] = math.cos(ax)  # 0

    ry[0, 0] = math.cos(ay)
    ry[0, 2] = -math.sin(ay)
    ry[2, 0] = math.sin(ay)
    ry[2, 2] = math.cos(ay)

    rz[0, 0] = math.cos(az)  # 0
    rz[0, 1] = -math.sin(az)  # 0
    rz[1, 0] = math.sin(az)  # 0
    rz[1, 1] = math.cos(az)  # 0

    r = rx.dot(ry).dot(rz)  # if we remove the lines we put    r=ry
    final = proj3dto2d.dot(trans.dot(r.dot(proj2dto3d)))

    return cv2.warpPerspective(img, final, (img.shape[1], img.shape[0]), None, cv2.INTER_LINEAR
                               , cv2.BORDER_CONSTANT, (0, 0, 0))


def imageRotate(img, angle):
    size_reverse = np.array(img.shape[1::-1])  # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:, :2])
    size_new = MM @ size_reverse
    M[:, -1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))


def imageTranslate(destination, num_frame):
    num_des = len(destination)
    res = num_frame % num_des

    image_offset = []
    offset_x = 10
    offset_y = 0

    num_mid_des = int(num_frame / num_des)
    for j in range(len(destination)):
        if (j == len(destination) - 1) & (res != 0):
            num_mid_des = num_mid_des + res
        for i in range(num_mid_des):
            image_offset.append([offset_x, offset_y])
            if j == 0:
                offset_x += int(destination[j][0] / num_mid_des)
                offset_y += int(destination[j][1] / num_mid_des)
            else:
                offset_x += int((destination[j][0] - destination[j - 1][0]) / num_mid_des)
                offset_y += int((destination[j][1] - destination[j - 1][1]) / num_mid_des)

    return image_offset


def centerPoint(marker_num):
    if marker_num == 1:
        return [[random.randint(110, 540), random.randint(110, 350)]], 0
    elif marker_num == 2:
        layout = random.randint(0, 1)
        if layout == 0:  # Left and right
            return [[random.randint(110, 220), random.randint(110, 350)],
                    [random.randint(420, 540), random.randint(110, 350)]], 1
        else:  # Up and down
            return [[random.randint(110, 540), random.randint(110, 170)],
                    [random.randint(110, 540), random.randint(310, 370)]], 2
    # elif marker_num == 3:
    #     layout = random.randint(0, 3)
    #     if layout == 0:  # Left 1 right 2
    #         return [[random.randint(110, 220), random.randint(110, 350)],
    #                 [random.randint(420, 540), random.randint(110, 190)],
    #                 [random.randint(420, 540), random.randint(290, 350)]]
    #     elif layout == 1:  # Left 2 right 1
    #         return [[random.randint(110, 220), random.randint(110, 190)],
    #                 [random.randint(110, 220), random.randint(290, 350)],
    #                 [random.randint(420, 540), random.randint(110, 350)]]
    #     elif layout == 2:  # Up 1 down 2
    #         return [[random.randint(110, 540), random.randint(110, 190)],
    #                 [random.randint(110, 220), random.randint(290, 350)],
    #                 [random.randint(420, 540), random.randint(290, 350)]]
    #     else:  # Up 2 down 1
    #         return [[random.randint(110, 220), random.randint(110, 190)],
    #                 [random.randint(420, 540), random.randint(110, 190)],
    #                 [random.randint(110, 540), random.randint(290, 350)]]
    # else:
    #     return [[random.randint(300, 560), random.randint(190, 340)],
    #             [random.randint(1160, 1620), random.randint(190, 340)],
    #             [random.randint(300, 560), random.randint(740, 880)],
    #             [random.randint(1160, 1620), random.randint(740, 880)]]

def random_choose(limit, range):
    a = random.randint(limit, range)
    b = random.randint(-range, -limit)
    return random.choice([a, b])

# Function to generate frames for the video
def generate_frames():
    # Set video properties
    width, height = 640, 480
    fps = 37
    duration_seconds = 2
    num_frames = fps * duration_seconds
    trans_limit = 100
    rotate_limit = 120
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc2 = cv2.VideoWriter_fourcc(*'mp4v')
    for num in range(0, 1):

        out = cv2.VideoWriter('DataGenOutput\\v4\\m' + str(num) + '.mp4', fourcc, fps, (width, height))
        out_label = cv2.VideoWriter('DataGenOutput\\v4\\l' + str(num) + '.mp4', fourcc2, fps,
                                    (width, height))
        black_bg = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(50):
            frame = black_bg.copy()
            frame_label = black_bg.copy()
            out.write(frame)
            out_label.write(frame_label)

        last_marker_num = 0
        marker_num = 0
        for video_num in range(5):
            while marker_num == last_marker_num:
                marker_num = random.randint(1, 2)
            last_marker_num = marker_num
            print(marker_num)
            markers, labels = [], []
            centers, modes = centerPoint(marker_num)
            rotate_init = []
            rotate_angle = []
            offset = []

            for _ in range(marker_num):
                index = random.randint(0, 29)
                thd_rotate = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-20, 20)]
                rotate_init.append(thd_rotate)
                resize = random.randint(100, 250)
                if marker_num == 3:
                    resize = random.randint(150, 200)

                random_motion = random.randint(0, 2)  # 0 only translate, 1 only rotate, 2 both
                if random_motion == 0:
                    rotate_angle.append([0, 0, random.randint(-75, 75)])
                else:
                    rotate_angle.append([random.randint(-15, 15), random.randint(-15, 15), random_choose(rotate_limit, 270)])
                trans_range = 180
                # if resize > 200:
                #     trans_range = 50
                # if marker_num == 1:
                #     resize = 300
                #     trans_range = 100

                if random_motion == 10:
                    offset.append([0, 0])
                else:
                    desnum = random.randint(3, 6)
                    des = []
                    for _ in range(desnum):
                        des.append([random_choose(trans_limit, trans_range), random_choose(trans_limit, trans_range)])
                    offset.append(imageTranslate(des, num_frames))

                marker = cv2.imread("DataGenInput/markerv1/marker" + str(index) + ".jpg")
                marker = imgZRotate(marker, thd_rotate[0], thd_rotate[1], thd_rotate[2])
                marker = cv2.resize(marker, (resize, resize))
                markers.append(marker)

                label = cv2.imread("DataGenInput/markerv1/label.jpg")
                label = imgZRotate(label, thd_rotate[0], thd_rotate[1], thd_rotate[2])
                label = cv2.resize(label, (resize, resize))
                labels.append(label)

            # Generate frames
            for i in range(num_frames):

                # Create a black frame
                frame = black_bg.copy()
                frame_label = black_bg.copy()

                for j in range(len(markers)):
                    bias = [320 - centers[j][0], 240 - centers[j][1]]
                    angle_x = (i / num_frames) * rotate_angle[j][0]
                    angle_y = (i / num_frames) * rotate_angle[j][1]
                    angle_z = (i / num_frames) * rotate_angle[j][2]
                    img = imgZRotate(markers[j], rotate_init[j][0] + angle_x, rotate_init[j][1] + angle_y,
                                     rotate_init[j][2] + angle_z)
                    lab = imgZRotate(labels[j], rotate_init[j][0] + angle_x, rotate_init[j][1] + angle_y,
                                     rotate_init[j][2] + angle_z)

                    bbox_x1 = int((width - img.shape[1]) / 2)
                    bbox_y1 = int((height - img.shape[0]) / 2)
                    bbox_x2 = bbox_x1 + img.shape[1]
                    bbox_y2 = bbox_y1 + img.shape[0]

                    xl = bbox_x1 - bias[0] + offset[j][i][0]
                    xr = bbox_x2 - bias[0] + offset[j][i][0]
                    yt = bbox_y1 - bias[1] + offset[j][i][1]
                    yd = bbox_y2 - bias[1] + offset[j][i][1]


                    if xl < 0:
                        correct = xl
                        xl += abs(correct)
                        xr += abs(correct)
                    if xr >= 640:
                        correct = xr - 639
                        xl -= correct
                        xr -= correct
                    if yt < 0:
                        correct = yt
                        yt += abs(correct)
                        yd += abs(correct)
                    if yd >= 480:
                        correct = yd - 479
                        yt -= correct
                        yd -= correct

                    frame[yt:yd, xl:xr] = img
                    frame_label[yt:yd, xl:xr] = lab

                # Write frame to video

                out.write(frame)
                out_label.write(frame_label)
        out.release()
        out_label.release()

def main():
    generate_frames()
    # test()

if __name__ == '__main__':
    main()
