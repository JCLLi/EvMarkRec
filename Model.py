import keras


def loss(lab_val, prd_val):
    pr = prd_val[:, :, :, :1]
    loss = keras.backend.mean(keras.backend.square(pr - lab_val))
    return loss


def get_model(img_size, init_filter, sub_block, expansion_factor):
    input = keras.Input(shape=img_size + (1,))

    x = keras.layers.Conv2D(init_filter, 7, strides=1, padding="same")(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(init_filter, 7, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    filter_num = init_filter * expansion_factor
    x = keras.layers.Conv2D(filter_num, 5, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    for count in range(sub_block):  # res block, sub-blocks
        res = x

        x = keras.layers.Conv2D(filter_num, 3, strides=1, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        x = keras.layers.Conv2D(filter_num, 3, strides=1, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.add([x, res])
        x = keras.layers.Activation("relu")(x)

        x = keras.layers.Conv2D(filter_num, 3, strides=1, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        filter_num *= expansion_factor

        x = keras.layers.Conv2D(filter_num, 3, strides=2, padding="same")(x)

    for count in range(sub_block):
        res = x

        x = keras.layers.Conv2DTranspose(filter_num, 3, strides=1, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        x = keras.layers.Conv2DTranspose(filter_num, 3, strides=1, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.add([x, res])
        x = keras.layers.Activation("relu")(x)

        x = keras.layers.Conv2DTranspose(filter_num, 3, strides=1, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        filter_num //= expansion_factor

        x = keras.layers.Conv2DTranspose(filter_num, 3, strides=2, padding="same")(x)

    filter_num //= expansion_factor
    x = keras.layers.Conv2DTranspose(filter_num, 5, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    filter_num //= expansion_factor
    x = keras.layers.Conv2DTranspose(filter_num, 7, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2DTranspose(filter_num, 7, strides=1, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    output = keras.layers.Conv2DTranspose(filter_num, 7, strides=1, padding="same")(x)

    model = keras.Model(input, output)
    return model
