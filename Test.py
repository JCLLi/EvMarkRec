import os

import keras
import keras.optimizers
import tensorflow as tf
import Model
import BacthGeneration as bg
import cv2
import numpy as np
import time


def test():
    # model = tf.keras.models.load_model("./CPbackup/80it_lrde_7f_2sub/080-0.0001-0.0012.keras", compile=False)
    model = tf.keras.models.load_model("./Dataset/080-0.0005-0.0027.keras")
    # model.compile(optimizer=keras.optimizers.Adam(9.0718e-05), loss=lambda y_true, y_pred: Model.loss(y_true, y_pred))
    # image_path = 'Dataset/Marker/Image/test'
    image_path = './Dataset/Test'
    # image_path = 'Dataset/Marker/Image/v2/validation'
    # label_path = 'Dataset/Marker/Label/v2/validation'
    ImageSize = (240, 320)
    ImagePath = [
        os.path.join(image_path, fname)
        for fname in os.listdir(image_path)
        if fname.endswith(".png")
    ]
    sorted_image_paths = sorted(ImagePath)
    images = []
    for path in sorted_image_paths:
        image = tf.io.read_file(path)
        image = tf.io.decode_png(image, channels=1)
        image = tf.image.resize(image, ImageSize)
        image = tf.image.convert_image_dtype(image, "float32") / 255.0
        image = tf.expand_dims(image, axis=0)
        images.append(image)

    # LabelPath = [
    #     os.path.join(label_path, fname)
    #     for fname in os.listdir(label_path)
    #     if fname.endswith(".jpg")
    # ]
    # labels = []
    # for path in LabelPath:
    #     label = tf.io.read_file(path)
    #     # image = tf.io.read_file("./Test/7.png")
    #     # image = tf.io.read_file("Dataset/Marker/Image/New\\42.jpg")
    #     label = tf.io.decode_png(label, channels=1)
    #     label = tf.image.resize(label, ImageSize)
    #     label = tf.image.convert_image_dtype(label, "float32") / 255.0
    #     label = tf.expand_dims(label, axis=0)
    #     labels.append(label)

    for i in range(len(images)):
        start_time = time.perf_counter()
        res = model(images[i])
        end_time = time.perf_counter()
        print("time " + str(end_time - start_time) + " seconds")

        res = res[:, :, :, :1]
        # res = res[..., 1]
        res = np.squeeze(res, axis=0)
        result = res > 0.5
        # IoU = keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)
        # IoU = keras.metrics.IoU(num_classes=2, target_class_ids=[1])
        # IoU.update_state(result, labels[i])
        # print("IoU: ", IoU.result().numpy())
        min_val = np.min(res)
        max_val = np.max(res)
        rescaled_array = (255 * (res - min_val) / (max_val - min_val)).astype(np.uint8)

        cv2.imwrite("Testres/" + str(i) + ".png", rescaled_array)

    # img = keras.utils.array_to_img(res)
    # res = (np.clip(res, 0., 1.) * 255.).astype(np.uint8)
    # display(rescaled_array)




def ntest():
    ImageSize = (240, 320)

    dataset = bg.get_dataset(20, 'Dataset/Marker/Image/test', 'Dataset/Marker/Label/test', ImageSize)
    img = []
    lab = []

    for image, label in dataset.take(len(dataset)):
        img.append(image)
        lab.append(label)
    img = tf.concat(img, axis=0)
    lab = tf.concat(lab, axis=0)
    ResNet = tf.keras.models.load_model(".\\Checkpoint", compile=False)
    ResNet.compile(optimizer=keras.optimizers.Adam(1e-4), loss=lambda y_true, y_pred: Model.loss(y_true, y_pred),  metrics=['accuracy'])
    loss, accuracy = ResNet.evaluate(img, lab)

    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
def main():
    test()

if __name__ == '__main__':
    main()