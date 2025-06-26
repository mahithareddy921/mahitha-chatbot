from extract_resume import extract_text_from_pdf
from pathlib import Path

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    pdf_paths = [Path("resume.pdf"), Path("mahitha.pdf")]
    all_text = ""

    for path in pdf_paths:
        if not path.exists():
            print(f"❌ File not found: {path}")
        else:
            print(f"📄 Extracting from: {path}")
            all_text += extract_text_from_pdf(path) + "\n"

    chunks = chunk_text(all_text, chunk_size=200)
    print(f"✅ Total chunks: {len(chunks)}\n")
    print("--- First chunk ---\n")
    print(chunks[0])
    if len(chunks) > 2:
        print("--- Third chunk ---\n")
        print(chunks[2])

