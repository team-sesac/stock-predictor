from utils import mkdir

def init(settings):
    set_data, set_hyper = settings["data"], settings["hyper"]
    mkdir(set_data["result_path"])