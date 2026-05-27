
import modal
import os

app = modal.App("dataset-inspector")
volume = modal.Volume.from_name("gemma4-sft-volume")

@app.function(
    image=modal.Image.debian_slim().pip_install("pandas", "pyarrow"),
    volumes={"/data": volume}
)
def inspect():
    import pandas as pd
    files = ["/data/data/indicvoices_r.parquet", "/data/data/kathbath.parquet", "/data/data/rasa.parquet"]
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_parquet(f)
                print(f"\n--- {f} ---")
                print(f"Columns: {df.columns.tolist()}")
                print(f"Head: {df.head(1).to_dict()}")
            except Exception as e:
                print(f"Error {f}: {e}")
        else:
            print(f"Not found: {f}")

if __name__ == "__main__":
    with app.run():
        inspect.remote()
