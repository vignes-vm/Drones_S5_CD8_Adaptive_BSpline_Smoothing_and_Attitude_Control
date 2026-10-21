import pypdf
import sys

reader = pypdf.PdfReader("bspline1.pdf")
print(f"Total pages: {len(reader.pages)}")

keywords = ["eq. 4", "equation 4", "fdyn", "f_dyn", "feasibility", "dynamic"]

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    for kw in keywords:
        if kw in text.lower():
            print(f"Page {i+1} matches '{kw}':")
            # print surrounding lines
            lines = text.split("\n")
            for line in lines:
                if kw in line.lower():
                    clean_line = line.strip().encode('ascii', 'ignore').decode('ascii')
                    print("  ", clean_line)
