import os

def get_new_file_counter(emoji):
    datapath = os.path.join(os.getcwd(), "data", emoji)
    maxcount = 0
    for path in os.listdir(datapath):
        parts = path.split(".")
        count = int(parts[0])
        maxcount = max(count, maxcount)
    
    return maxcount

