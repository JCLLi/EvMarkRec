import random

import cv2
import numpy as np
import os

def add_noise(gray_frame):
    read = False
    while not read:
        index = random.randint(11, 4656)
        index = "{:010d}".format(index)
        noise = cv2.imread('.\\DataGenOutput\\v2\\noise\\_' + str(index) + '.png')
        if noise is not None:
            read = True

    noise = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
    noise = cv2.resize(noise, (346, 260))
    mask = cv2.inRange(gray_frame, 110, 140)

    # Invert the mask
    mask = cv2.bitwise_not(mask)

    kernel = np.ones((5, 5), np.uint8)

    dilated_mask = cv2.dilate(mask, kernel, iterations=3)

    # Combine the two images using the mask
    result = cv2.bitwise_and(gray_frame, gray_frame, mask=dilated_mask)
    result += cv2.bitwise_and(noise, noise, mask=cv2.bitwise_not(dilated_mask))
    return result


def v2i_image():
    frame_count = 0
    cap = cv2.VideoCapture('.\\DataGenOutput\\v2\\event\\mv.avi')
    cap_label = cv2.VideoCapture('.\\DataGenOutput\\v2\\event\\lvs.mp4')

    # Read and process each frame
    while frame_count >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        cap_label.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        _, label_frame = cap_label.read()
        if not ret:
            break  # Break the loop if no more frames are available

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_label = cv2.cvtColor(label_frame, cv2.COLOR_BGR2GRAY)

        color = gray_frame[0, 0]
        is_one_color = (gray_frame == color).all()

        num = 0
        stride = 10
        if not is_one_color:
            for row in gray_frame:
                for pixel in row:
                    if pixel != color:
                        num += 1

        color_perc = num / (346 * 260)

        if (frame_count % 499 <= 10) or (frame_count % 499 >= 490):
            record = 0
        else:
            record = 1

        if ((not is_one_color) and (color_perc > 0.00006)) and (record == 1):
            # Save the noisy grayscale frame as an image
            index = "{:010d}".format(int(frame_count / stride))
            output_path = os.path.join('.\\Dataset\\Marker\\Image\\v2\\validation', f'{index}.jpg')
            output_path_label = os.path.join('.\\Dataset\\Marker\\Label\\v2\\validation', f'{index}.jpg')

            gray_frame = add_noise(gray_frame)
            cv2.imwrite(output_path, gray_frame)
            cv2.imwrite(output_path_label, gray_frame_label)
            # af += 1
        frame_count += stride

    # Release the video capture object
    cap.release()
    print(f"Processed {frame_count} frames.")


def noise_generate():
    frame_count = 0

    cap = cv2.VideoCapture('.\\DataGenOutput\\test\\V2E\\noise\\tennis.avi')

    # Read and process each frame
    a = 0
    while a == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no more frames are available

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color = gray_frame[0, 0]

        # Add random noise to the grayscale frame
        for _ in range(500):  # Adjust the number of pixels added
            x = np.random.randint(1, gray_frame.shape[1])
            y = np.random.randint(1, gray_frame.shape[0])
            if np.random.rand() < 0.5:
                gray_frame[y, x] = 0  # Set pixel to black with 50% probability
            else:
                gray_frame[y, x] = 255

        # Save the noisy grayscale frame as an image
        output_path = os.path.join('.\\DataGenOutput\\test\\V2E\\noise\\images',
                                   f'{frame_count + 443 + 555 + 558 + 557 + 554}.jpg')
        cv2.imwrite(output_path, gray_frame)

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Processed {frame_count} frames.")




def main():
    v2i_image()

if __name__ == '__main__':
    main()
