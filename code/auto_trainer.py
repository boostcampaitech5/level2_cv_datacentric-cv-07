import os

BASE_PATH = "./"
CONFIG_PATH = os.path.join(BASE_PATH, "configs")

CONFIG_QUEUE_PATH = os.path.join(CONFIG_PATH, "queue")
CONFIG_ENDS_PATH = os.path.join(CONFIG_PATH, "ends")

configs = [file for file in os.listdir(CONFIG_QUEUE_PATH) if file != ".gitkeep"]

if configs:
    print(f"total config file: {configs}")
    # train.py
    print(f"current train config file: {configs[0]}")
    os.system(f"python train.py --config {os.path.join(CONFIG_QUEUE_PATH, configs[0])}")
    
    # inference.py
    print(f"current inference config file: {configs[0]}")
    os.system(f"python inference.py --config {os.path.join(CONFIG_QUEUE_PATH, configs[0])}")
    
    # config 옮기기
    os.system(f"mv {os.path.join(CONFIG_QUEUE_PATH, configs[0])} {os.path.join(CONFIG_ENDS_PATH, configs[0])}")