from utils import mkdir

def init(settings):
    set_data, set_hyper = settings["data"], settings["hyper"]
    platform = set_data["platform"]
    info = set_data["info"][platform]
    base_path = info["base_path"]
    result_path = info["result_path"]

    train_data = base_path + set_data["filename"]
    test_data = base_path + set_data["test_filename"]
    target_data = base_path + set_data["target_filename"]
    mkdir(result_path)
    
    if platform == "colab":

        DIR_PATH = info["drive_path"]
        assert DIR_PATH is not None, "[!] Enter the foldername."
        
        # Change dariectory to current folder
        root_path = info["root_path"]
        base_path = root_path + DIR_PATH + base_path
        result_path = root_path + DIR_PATH + result_path
        # return platform, base_path, result_path
        print(f"base_path = {base_path}")
        print(f"result_path = {result_path}")

    elif platform == "kaggle":
        test_path = info["test_path"]
        test_data = base_path + test_path + set_data["test_filename"]
        target_data = base_path + test_path + set_data["target_filename"]
    return platform, result_path, (train_data, test_data, target_data)