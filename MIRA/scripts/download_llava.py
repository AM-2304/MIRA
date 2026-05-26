
import modal
import os

app = modal.App("download-llava-images")
image = modal.Image.debian_slim().apt_install("curl", "unzip")
volume = modal.Volume.from_name("gemma4-sft-volume")

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=86400, # 24 hours
)
def download():
    import subprocess
    
    img_dir = "/data/data/images"
    os.makedirs(img_dir, exist_ok=True)
    
    # Download COCO 2017 training images (standard for LLaVA-150k)
    url = "http://images.cocodataset.org/zips/train2017.zip"
    zip_path = "/data/train2017.zip"
    
    print(f"Starting download of {url}...")
    subprocess.run(["curl", "-L", url, "-o", zip_path], check=True)
    
    print("Unzipping images...")
    subprocess.run(["unzip", "-q", zip_path, "-d", img_dir], check=True)
    
    print("Cleaning up zip...")
    os.remove(zip_path)
    
    volume.commit()
    print("Download and extraction complete.")

if __name__ == "__main__":
    with app.run():
        download.remote()
