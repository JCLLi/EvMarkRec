import cv2
import numpy as np
import random

def ring_buffer(i, size):
    if i > size - 1:
        return i - size
    elif i < 0:
        return i + size
    else:
        return i

def twoD_array(matrix, row, col, value):
    # Extend the rows if necessary
    while len(matrix) <= row:
        matrix.append([])

    # Extend the columns if necessary
    while len(matrix[row]) <= col:
        matrix[row].append(0)  # Assuming you want to fill missing elements with 0

    # Set the value at the specified position
    matrix[row][col] = value


def show_pic(pic):
    cv2.imshow("res", pic)
    cv2.waitKey()
    cv2.destroyAllWindows()


def image_smooth(image):
    if image is not None:
        result_image = np.full_like(image, 128)  # Initialize with gray (128)

        # Get the dimensions of the image
        height, width = image.shape

        # Define the offsets for the 3x3 neighborhood
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]

        # Iterate through the image (excluding the border pixels)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Get the pixel value
                center_pixel = image[y, x]

                # Check if the center pixel is black or white
                if center_pixel <= 60 or center_pixel >= 196:
                    # Get the 3x3 neighborhood
                    neighborhood = [image[y + dy, x + dx] for dy, dx in offsets]
                    count = 0
                    for pixel in neighborhood:
                        if (int(pixel) + int(center_pixel)) < 160 or (int(pixel) + int(center_pixel)) > 352:
                            count += 1
                    if count >= 6:
                        result_image[y, x] = center_pixel

        return result_image
        # cv2.imwrite("./frames/smoothed/" + str(index) + ".png", result_image)


def cal_gradients(index):
    path = "./frames/Test2/res/scale1.3/" + str(index) + ".png"
    image = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    if image is not None:
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150)

        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

        # Normalize magnitude from 0 to 255
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        avg_angle = np.mean(angle)
        angle_8bit = np.uint8(255 * avg_angle / 360)
        return angle_8bit
    else:
        return 0



def line_seg(index):
    path = "./frames/Test2/res/scale1.3/" + str(index) + ".png"
    prd_res = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    if prd_res is not None:

        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = lsd.detect(prd_res)[0]
        img = lsd.drawSegments(prd_res, lines)

        cv2.imwrite("./frames/Test2/res/ls/" + str(index) + ".png", img)
        r, c = calculate_line_lengths(prd_res, lines)
        return int(r), int(c)
    else:
        return 0, 0

def divide(image, index):
    if image is not None:
        _, white_mask = cv2.threshold(image, 148, 255, cv2.THRESH_BINARY)
        _, black_mask = cv2.threshold(image, 108, 255, cv2.THRESH_BINARY_INV)

        dark_gray_image = np.full_like(image, 255)

        white_parts = cv2.bitwise_and(image, dark_gray_image, mask=white_mask)
        black_parts = cv2.bitwise_or(image, dark_gray_image, mask=black_mask)

        cv2.imwrite("./frames/Test3/res/on/" + str(index) + ".png", white_parts)
        cv2.imwrite("./frames/Test3/res/off/" + str(index) + ".png", black_parts)


def calculate_line_lengths(img, lines):

    # Initialize sum of lengths for horizontal and vertical lines
    horizontal_length = 0
    vertical_length = 0

    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = line[0]
            aa = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            cv2.line(aa, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 3)

            # Calculate the length of the line segment
            length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

            # Calculate the angle of the line
            angle = abs(np.arctan2((y1 - y0), (x1 - x0)))
            # print(angle)
            # Check if the line is horizontal or vertical
            if angle > np.pi * 17 / 9 or angle < np.pi / 9 or (np.pi * 8 / 9 < angle < np.pi * 10 / 9):
                horizontal_length += length
                # print("row: " + str(int(length)))
            elif (np.pi * 3.5 / 9 < angle < np.pi * 5.5 / 9) or (np.pi * 12.5 / 9 < angle < np.pi * 14.5 / 9):
                vertical_length += length
                # print("col: " + str(int(length)))
            # Decode.show_pic(aa)
    return horizontal_length, vertical_length


def generate_valid_number():
    while True:
        # Generate a random 14-bit number
        num = random.randint(0, 2 ** 14 - 1)

        # Convert number to binary and pad to 14 bits
        bits = f'{num:014b}'

        # Convert the binary string to a list of integers (bits)
        bit_list = [int(bit) for bit in bits]

        # Check constraints
        if not (bit_list[0] != 0 and bit_list[1] != 0 and bit_list[2] != 0):  # bits 0, 1, 2
            if not (bit_list[2] != 0 and bit_list[6] != 0 and bit_list[10] != 0):  # bits 2, 6, 10
                if not (bit_list[3] != 0 and bit_list[7] != 0 and bit_list[11] != 0):  # bits 3, 7, 11
                    if not (bit_list[11] != 0 and bit_list[12] != 0 and bit_list[13] != 0):  # bits 11, 12, 13
                        print(bits)
                        return bit_list


def generate_marker():
    for i in range(31):
        img = cv2.imread("prototype.png", cv2.IMREAD_GRAYSCALE)
        number = generate_valid_number()
        for y in range(4):
            for x in range(4):
                if x == 0 and y == 0:
                    continue
                elif x == 3 and y == 3:
                    continue
                else:
                    number_index = y * 4 + x - 1
                    a = number_index
                    if number[a] == 1:
                        p1 = (30 + x * 12, 30 + y * 12)
                        p2 = (41 + x * 12, 30 + y * 12)
                        p3 = (41 + x * 12, 41 + y * 12)
                        p4 = (30 + x * 12, 41 + y * 12)
                        pts = np.array([p1, p2, p3, p4], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.fillPoly(img, [pts], 255)
        # show_pic(img)
        center = (img.shape[1] // 2, img.shape[0] // 2)
        angle = random.randint(0, 360)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 0.7)

        # Perform the rotation
        # rotated_image = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
        cv2.imwrite("./random_message/" + str(i) + ".png", img)

