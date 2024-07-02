import glob
import os
import keras.backend
import keras.optimizers
import keras.callbacks
import keras.metrics
import tensorflow as tf
import BacthGeneration as Bg
import Model
import IoU

def train(setting):
    # GPU setting
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 17) * 0.9)]
            )
        except RuntimeError as e:
            print(e)

    # Load train and validate dataset
    train_dataset = Bg.get_dataset(setting.batch_size, setting.train_image_path, setting.train_label_path,
                                   setting.image_size)
    validate_dataset = Bg.get_dataset(setting.batch_size, setting.validate_image_path, setting.validate_label_path,
                                      setting.image_size)

    # Load model
    if setting.trained_epoch == 0:
        network = Model.get_model3(setting.image_size, setting.init_filter,
                                  setting.sub_block, setting.expansion_factor)
    else:
        print("Loading latest model from file:", setting.latest_model_file)
        network = tf.keras.models.load_model(setting.latest_model_file, compile=False)
        

    # Compile model
    network.compile(optimizer=keras.optimizers.Adam(learning_rate=setting.scheduler),
                    loss=setting.loss,
                    metrics=[IoU.CustomIoU()])
                    # metrics=[keras.metrics.IoU(num_classes=2, target_class_ids=[0, 1])])
                    # metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)])

    custom_iou_callback = IoU.CustomIoUCallback()
    # Checkpoint and tensor board setup
    check_point = keras.callbacks.ModelCheckpoint(
        filepath=setting.checkpoint_path + "{epoch:03d}-{loss:.4f}.keras",
        save_weights_only=False,  # Save entire model
        save_best_only=False,
        save_freq= 228 * setting.checkpoint_epoch,  # Save at the end of each epoch 9120
        verbose=1)

    tensor_board = keras.callbacks.TensorBoard(log_dir="Log", histogram_freq=0, write_graph=False, write_images=False,
                                               write_steps_per_second=False, update_freq="epoch")

    callbacks = [check_point, tensor_board, custom_iou_callback]

    # Summarize network architecture
    network.summary()
    print("Trained epoch is:", setting.trained_epoch)

    # Start training
    print("Continue?")
    user_input = input()
    if user_input == "yes" or user_input == "YES" or user_input == "Y" or user_input == "y":
        network.fit(train_dataset, epochs=setting.epochs, callbacks=callbacks, validation_data=validate_dataset,
                    shuffle=True, verbose=1, initial_epoch=setting.trained_epoch, validation_freq=5)
