
import modal
import os

app = modal.App("ls-indictts")
volume = modal.Volume.from_name("gemma4-sft-volume")

@app.function(volumes={"/data": volume})
def ls():
    for root, dirs, files in os.walk("/data/data/indictts"):
        print(f"Dir: {root}")
        for f in files[:5]: print(f"  {f}")

if __name__ == "__main__":
    with app.run():
        ls.remote()
