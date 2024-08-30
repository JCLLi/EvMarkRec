import os
import Setting
import keras
import keras.optimizers
import tensorflow as tf
import cv2
import numpy as np
import time
import IoU

def test():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 5) * 1)]
            )
        except RuntimeError as e:
            print(e)


    json_file = "./settings.json"
    setting = Setting.read_json(json_file)
    iteration = "220"
    filename = "0.0096"
    model = tf.keras.models.load_model("./Checkpoints/" + iteration + "-" + filename + ".keras", compile=False)
    # model = tf.keras.models.load_model("./Checkpoints/120-0.0006-0.0019.keras", compile=False)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=setting.scheduler),
                    loss=setting.loss,
                    metrics=[IoU.CustomIoU()])

    image_path = './Dataset/validation'

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

    # label_path = './Dataset/Marker/Label/v3/validation/'

    # ImageSize = (240, 320)
    # LabelPath = [
    #     os.path.join(label_path, fname)
    #     for fname in os.listdir(image_path)
    #     if fname.endswith(".png")
    # ]
    # sorted_label_paths = sorted(LabelPath)
    # labels = []
    # for path in sorted_label_paths:
    #     label = tf.io.read_file(path)
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
        res = np.squeeze(res, axis=0)
        if setting.loss == "mean_squared_error" or isinstance(setting.loss, keras.losses.BinaryCrossentropy):
            min_val = np.min(res)
            max_val = np.max(res)
            rescaled_array = (255 * (res - min_val) / (max_val - min_val)).astype(np.uint8)
            # iou = keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.4) 
            # iou.update_state(labels[i], rescaled_array)
            # print(str(i) + "IoU: " +  str(iou.result().numpy()))
            # iou.reset_state()
            # end_time = time.perf_counter()
            # print("time " + str(end_time - start_time) + " seconds")
            cv2.imwrite("Res/" + str(i) + ".png", rescaled_array)
        else:
            probabilities = 1 / (1 + np.exp(-res))
            binary_mask = (probabilities > 0.5).astype(np.uint8) * 255
            cv2.imwrite("Res/" + str(i) + ".png", binary_mask * 255)

def main():
    test()

if __name__ == '__main__':
    main()