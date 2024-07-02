import json
import Schecduler
import keras
import os
import glob

class Setting:
    def __init__(self, setting):
        self.image_size = tuple(setting["data"]["image_size"])
        self.train_image_path = setting["data"]["train_image_path"]
        self.train_label_path = setting["data"]["train_label_path"]
        self.validate_image_path = setting["data"]["validate_image_path"]
        self.validate_label_path = setting["data"]["validate_label_path"]

        self.checkpoint_path = setting["training"]["checkpoint_path"]
        
        self.trained_epoch = 0
        self.latest_model_file = ""
        if os.listdir(self.checkpoint_path):
            files = glob.glob(os.path.join(self.checkpoint_path, '*'))
            files.sort(key=os.path.getmtime, reverse=True)
            self.latest_model_file = files[0]
            self.trained_epoch = int((self.latest_model_file.split("/")[1]).split("-")[0])
        
        if setting["training"]["loss"] == "BCE":
            self.loss = keras.losses.BinaryCrossentropy()
        elif setting["training"]["loss"] == "MSE":
             self.loss = "mean_squared_error"
        
        if setting["training"]["learning_rate_scheduler"] == "Constant":
            self.scheduler = setting["training"]["Constant"]["default_learning_rate"]
        
        elif setting["training"]["learning_rate_scheduler"] == "ConCos":
            self.scheduler = Schecduler.ConCos(setting["training"]["ConCos"]["default_learning_rate"], 
                                    setting["training"]["ConCos"]["end_learning_rate"], 
                                    setting["training"]["ConCos"]["constant_epoch"] * 228, 
                                    setting["training"]["epochs"] * 228,
                                    self.trained_epoch * 228)

        elif setting["training"]["learning_rate_scheduler"] == "LinCos":
            self.scheduler = Schecduler.LinCos(setting["training"]["LinCos"]["default_learning_rate"], 
                                    setting["training"]["LinCos"]["end_learning_rate"], 
                                    setting["training"]["LinCos"]["warmup_epoch"] * 228, 
                                    setting["training"]["epochs"] * 228,
                                    self.trained_epoch * 228)

        elif setting["training"]["learning_rate_scheduler"] == "InverseTimeDecay":
            self.scheduler = keras.optimizers.schedules.InverseTimeDecay(
                                initial_learning_rate=setting["training"]["InverseTimeDecay"]["default_learning_rate"],
                                decay_steps=setting["training"]["InverseTimeDecay"]["decay_epoch"] * 228,
                                decay_rate=setting["training"]["InverseTimeDecay"]["decay_rate"])
            
        elif setting["training"]["learning_rate_scheduler"] == "CosineDecayRestarts":
            self.scheduler = keras.optimizers.schedules.CosineDecayRestarts(
                                initial_learning_rate=setting["training"]["CosineDecayRestarts"]["default_learning_rate"],
                                first_decay_steps=setting["training"]["CosineDecayRestarts"]["decay_epoch"] * 228,
                                t_mul=setting["training"]["CosineDecayRestarts"]["t_mul"],
                                m_mul=setting["training"]["CosineDecayRestarts"]["m_mul"])
            
        self.epochs = setting["training"]["epochs"]
        self.checkpoint_epoch = setting["training"]["checkpoint_epoch"]
        self.batch_size = setting["training"]["batch_size"]

        self.init_filter = setting["network"]["init_filter"]
        self.sub_block = setting["network"]["sub_block"]
        self.expansion_factor = setting["network"]["expansion_factor"]


def read_json(file_path):
    with open(file_path, 'r') as f:
        settings = json.load(f)
    return Setting(settings)


def create_json():
    setting = {
        "data": {
            "image_size": (240, 320),
            "train_image_path": "Dataset/Marker/Image/v2",
            "train_label_path": "Dataset/Marker/Label/v2",
            "validate_image_path": "Dataset/Marker/Image/v2/validation",
            "validate_label_path": "Dataset/Marker/Label/v2/validation",
        },
        "training": {
            "checkpoint_path": "./Checkpoints/",
            "default_learning_rate": 0.001,
            "lr_decay_threshold": 40,
            "batch_size": 20,
            "epochs": 10,
        },

        "network": {
            "init_filter": 32,
            "sub_block": 2,
            "expansion_factor": 2
        }
    }

    # Specify the filename for the JSON file
    json_file = "./settings.json"

    # Write the settings to the JSON file
    with open(json_file, 'w') as f:
        json.dump(setting, f, indent=4)
        # json.dump(training, f, indent=4)
        # json.dump(network, f, indent=4)

    print("Settings saved to", json_file)
