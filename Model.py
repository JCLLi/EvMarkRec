import keras

def ResBlock(input, filter_num, downsampling, cc): # cc: crop/conpensate
    res = input
    conv = keras.layers.Conv2D if downsampling else keras.layers.Conv2DTranspose
    x = keras.layers.BatchNormalization()(input)
    x = keras.layers.Activation("relu")(x)
    x = conv(filter_num, 3, strides=1, padding="same")(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = conv(filter_num, 3, strides=1, padding="same")(x)
    
    if cc:
        res = conv(filter_num, 1, strides=1, padding="same")(res)
        res = keras.layers.BatchNormalization()(res)

    x = keras.layers.add([x, res])
    return x


def ConvBlock(input, filter_num, filter_size, strides):
    x = keras.layers.Conv2D(filter_num, filter_size, strides, padding="same")(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


def ConvTransBlock(input, filter_num, filter_size, strides):
    x = keras.layers.Conv2DTranspose(filter_num, filter_size, strides, padding="same")(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


def get_model(img_size, init_filter, sub_block, expansion_factor): #v3
    input = keras.Input(shape=img_size + (1,))

    filter_num = init_filter
    c0 = ConvBlock(input, filter_num, 7, 1) #32

    filter_num *= expansion_factor
    c1 = ConvBlock(c0, filter_num, 7, 2)
    c2 = ConvBlock(c1, filter_num, 5, 1) #64
    c2 = ConvBlock(c2, filter_num, 5, 1) #64

    filter_num *= expansion_factor
    c3 = ConvBlock(c2, filter_num, 5, 2)
    r0 = ResBlock(c3, filter_num, True, False) #128

    filter_num *= expansion_factor
    d0 = ConvBlock(r0, filter_num, 3, 2) 

    r1 = ResBlock(d0, filter_num, True, False) #256
    x = keras.layers.Dropout(0.4)(r1)
    rt0 = ResBlock(x, filter_num, False, False) #256t
    x = keras.layers.Dropout(0.4)(rt0)

    filter_num //= expansion_factor
    u0 = ConvTransBlock(x, filter_num, 3, 2) 

    rt1 = ResBlock(u0, filter_num, False, False) #128t

    filter_num //= expansion_factor
    ct0 = ConvTransBlock(rt1, filter_num, 5, 2)
    ct1 = ConvTransBlock(ct0, filter_num, 5, 1) #64t
    ct1 = ConvTransBlock(ct1, filter_num, 5, 1) #64t

    filter_num //= expansion_factor
    ct2 = ConvTransBlock(ct1, filter_num, 7, 2)
    ct3 = ConvTransBlock(ct2, filter_num, 7, 1) #32t

    # filter_num //= expansion_factor
    output = keras.layers.Conv2DTranspose(1, 1, strides=1, padding="same", activation='sigmoid')(ct3)

    model = keras.Model(input, output)
    return model

