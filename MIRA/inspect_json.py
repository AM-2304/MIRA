
import modal
import json
import os

app = modal.App("inspect-json")
volume = modal.Volume.from_name("gemma4-sft-volume")

@app.function(volumes={"/data": volume})
def inspect():
    path = "/data/data/llava_instruct_150k.json"
    if not os.path.exists(path):
        print("Not found at /data/data/llava_instruct_150k.json")
        return
    with open(path, "r") as f:
        data = json.load(f)
        print(json.dumps(data[0], indent=2))

if __name__ == "__main__":
    with app.run():
        inspect.remote()
