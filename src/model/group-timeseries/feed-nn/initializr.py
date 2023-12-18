from utils import mkdir

def init(settings):
    set_data, set_hyper = settings["data"], settings["hyper"]
    platform = set_data["platform"]
    info = set_data["info"][platform]
    base_path = info["base_path"]
    result_path = info["result_path"]
    mkdir(result_path)
    
    if platform == "colab":
        from google.colab import drive
        drive.mount('/content/drive')

        DIR_PATH = ""
        assert DIR_PATH is not None, "[!] Enter the foldername."

        import sys
        sys.path.append(f"/content/drive/MyDrive/{DIR_PATH}")
        
        # Change dariectory to current folder
        # %cd /content/drive/MyDrive/$DIR_PATH
    
    return platform, base_path, result_path