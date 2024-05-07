import glob
import os
import keras.backend
import keras.optimizers
import keras.callbacks
import keras.metrics
import tensorflow as tf
import math
import BacthGeneration as Bg
import Model

lr_decay_threshold = 15


def scheduler(epoch, lr):
    if epoch > lr_decay_threshold and lr > 0.0001:
        return lr * math.exp(-0.1)
    elif lr < 0.0001:
        return lr
    else:
        return lr


def train(setting):
    # GPU setting
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 6) * 0.999)]
            )
        except RuntimeError as e:
            print(e)

    # Load train and validate dataset
    train_dataset = Bg.get_dataset(setting.batch_size, setting.train_image_path, setting.train_label_path,
                                   setting.image_size)
    validate_dataset = Bg.get_dataset(setting.batch_size, setting.validate_image_path, setting.validate_label_path,
                                      setting.image_size)

    trained_epoch = 0
    new_train = True
    latest_model_file = ""

    # Check the existence of checkpoints
    if os.listdir(setting.checkpoint_path):
        new_train = False
        files = glob.glob(os.path.join(setting.checkpoint_path, '*'))
        files.sort(key=os.path.getmtime, reverse=True)
        latest_model_file = files[0]
        print("Loading latest model from file:", latest_model_file)

    # Load model
    if new_train:
        network = Model.get_model(setting.image_size, setting.init_filter,
                                  setting.sub_block, setting.expansion_factor)
    else:
        network = tf.keras.models.load_model(latest_model_file, compile=False)
        trained_epoch = int((latest_model_file.split("\\")[1]).split("-")[0])

    # Learning rate initialization after loading model
    learning_rate = setting.default_learning_rate
    for i in range(trained_epoch):
        if i > setting.lr_decay_threshold:
            learning_rate *= math.exp(-0.1)

    # Compile model
    network.compile(optimizer=keras.optimizers.Adam(learning_rate),
                    loss="mean_squared_error",
                    metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)])

    # Checkpoint, dynamic learning rate and tensor board setup
    check_point = keras.callbacks.ModelCheckpoint(
        filepath=setting.checkpoint_path + "{epoch:03d}-{loss:.4f}-{val_loss:.4f}.keras",
        save_weights_only=False,  # Save entire model
        save_best_only=False,
        save_freq='epoch',  # Save at the end of each epoch
        verbose=1)
    dynamic_lr = keras.callbacks.LearningRateScheduler(scheduler)
    tensor_board = keras.callbacks.TensorBoard(log_dir="Log", histogram_freq=0, write_graph=False, write_images=False,
                                               write_steps_per_second=False, update_freq="epoch")

    callbacks = [check_point, tensor_board, dynamic_lr]

    # Summarize network architecture
    network.summary()
    print("Trained epoch is:", trained_epoch)
    print("Learning rate is:", learning_rate)
    # visualkeras.layered_view(ResNet, to_file='./output.png').show()

    # Start training
    print("Continue?")
    user_input = input()
    if user_input == "yes" or user_input == "YES" or user_input == "Y" or user_input == "y":
        network.fit(train_dataset, epochs=setting.epochs, callbacks=callbacks, validation_data=validate_dataset,
                    shuffle=True, verbose=1, initial_epoch=trained_epoch)
