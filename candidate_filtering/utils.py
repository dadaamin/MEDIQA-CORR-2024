import json

def load_json(path):
    with open(path) as f:
        return json.load(f)

def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)