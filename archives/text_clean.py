# import pandas as pd
# text = pd.read_csv(r"C:\\Users\\gopal\\Downloads\\archive (1)\\train_essays_RDizzl3_seven_v2.csv")
# print(text.head())
# text = text["text"]
# print(text.head())
# data = " ".join(text.astype(str))
# with open ("traininig.txt","w",encoding="utf-8") as f:
#     f.write(data)

with open ("training.txt","r",encoding="utf-8") as f:
    text = f.read()
import re
import unicodedata
def clean(text:str) ->str:
    text = unicodedata.normalize("NFC",text)
    text = re.sub(r"<[^>]+>","",text)
    text = text.replace("\u200b","")
    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if char in "\n\t":
            cleaned.append(char)
        elif not category.startswith("C"):
            cleaned.append(char)
    text = "".join(cleaned)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text

text = clean(text)
with open ("training.txt","w",encoding="utf-8") as f:
    f.write(text)
    