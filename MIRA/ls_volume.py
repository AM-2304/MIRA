
import modal
import os

app = modal.App("ls-volume")
volume = modal.Volume.from_name("gemma4-sft-volume")

@app.function(volumes={"/data": volume})
def ls():
    output = []
    import os
    for root, dirs, files in os.walk("/data"):
        depth = root.replace("/data", "").count(os.sep)
        if depth > 2: continue
        output.append(f"Dir: {root}")
        for f in files:
            try:
                size = os.path.getsize(os.path.join(root, f)) / 1e6
                output.append(f"  {f} ({size:.2f} MB)")
            except Exception as e:
                output.append(f"  {f} (error: {e})")
    return "\n".join(output)

if __name__ == "__main__":
    with app.run():
        print(ls.remote())
