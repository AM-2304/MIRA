import sys
try:
    import pypdf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    import pypdf

def extract_pdf(pdf_path, txt_path):
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"--- Page {i+1} ---\n"
        text += page.extract_text() + "\n"
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted {len(reader.pages)} pages to {txt_path}")

if __name__ == "__main__":
    extract_pdf("/Users/vasu/Documents/GitHub/gemma4-ira-companion/Multimodal IRA task.pdf", "/Users/vasu/Documents/GitHub/gemma4-ira-companion/pdf_extracted.txt")
