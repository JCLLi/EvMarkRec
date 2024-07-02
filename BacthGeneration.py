import os
import tensorflow as tf



def get_dataset(batch_size, image_path, labels_path, image_size):
    ImagePath = sorted([
        os.path.join(image_path, fname)
        for fname in os.listdir(image_path)
        if fname.endswith(".png")
    ])
    LablePath = sorted([
        os.path.join(labels_path, fname)
        for fname in os.listdir(labels_path)
        if fname.endswith(".png")
    ])

    def normalize(image_name, label_name):
        image = tf.io.read_file(image_name)
        image = tf.io.decode_png(image, channels=1)
        image = tf.image.resize(image, image_size)
        image = tf.image.convert_image_dtype(image, "float32") / 255.0

        label = tf.io.read_file(label_name)
        label = tf.io.decode_png(label, channels=1)
        label = tf.image.resize(label, image_size)
        label = tf.image.convert_image_dtype(label, "float32") / 255.0

        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((ImagePath, LablePath))
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=len(ImagePath))

    # Repeat the dataset indefinitely for epochs
    return dataset.batch(batch_size)