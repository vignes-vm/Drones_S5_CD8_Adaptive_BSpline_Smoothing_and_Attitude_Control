import pypdf

reader = pypdf.PdfReader("bspline1.pdf")
page = reader.pages[4] # Page 5 is index 4
text = page.extract_text()
clean_text = text.encode('ascii', 'ignore').decode('ascii')
print(clean_text)
