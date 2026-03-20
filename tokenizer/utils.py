import json

def read_data(file):
    with open(file, "r", encoding = "utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    
def save(obj, path):
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(obj, f, indent =2)
    
def load(path):
    with open(path, "r", encoding = "utf-8") as f:
        return json.load(f)
