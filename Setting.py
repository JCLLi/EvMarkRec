import json


class Setting:
    def __init__(self, setting):
        self.image_size = tuple(setting["data"]["image_size"])
        self.train_image_path = setting["data"]["train_image_path"]
        self.train_label_path = setting["data"]["train_label_path"]
        self.validate_image_path = setting["data"]["validate_image_path"]
        self.validate_label_path = setting["data"]["validate_label_path"]

        self.checkpoint_path = setting["training"]["checkpoint_path"]
        self.default_learning_rate = setting["training"]["default_learning_rate"]
        self.lr_decay_threshold = setting["training"]["lr_decay_threshold"]
        self.batch_size = setting["training"]["batch_size"]
        self.epochs = setting["training"]["epochs"]

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
            "checkpoint_path": "Checkpoints/",
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
