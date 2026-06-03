# check_labels.py — run this BEFORE the conversion script
import xml.etree.ElementTree as ET
import os
from collections import Counter

labels = Counter()
for f in os.listdir("annotations"):
    if f.endswith(".xml"):
        root = ET.parse(f"annotations/{f}").getroot()
        for obj in root.findall("object"):
            labels[obj.find("name").text.strip()] += 1

print("Labels found in your dataset:")
for label, count in labels.most_common():
    print(f"  '{label}': {count} objects")