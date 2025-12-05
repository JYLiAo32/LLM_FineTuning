import docx
import glob

texts = []

for file in glob.glob("data/*.docx"):
    doc = docx.Document(file)
    content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    texts.append({"file": file, "content": content})

print(texts[0])
