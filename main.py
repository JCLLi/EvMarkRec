import Train
import Setting


def main():
    json_file = "./settings.json"
    setting = Setting.read_json(json_file)
    Train.train(setting)


if __name__ == '__main__':
    main()
