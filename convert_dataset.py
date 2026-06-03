# convert_dataset.py  — FIXED VERSION
import os
import xml.etree.ElementTree as ET
import shutil
import random

IMAGES_DIR      = "images"
ANNOTATIONS_DIR = "annotations"
OUTPUT_DIR      = "NEWDS"
TRAIN_RATIO     = 0.8

CLASS_MAP = {
    # BikesHelmets dataset labels
    "With Helmet":    0,
    "Without Helmet": 1,
    # hard_hat_workers dataset labels
    "helmet":         0,   # same as "With Helmet"
    "head":           1,   # same as "Without Helmet" (bare head = no helmet)
    # "person" is intentionally excluded — full-body bbox not useful for helmet detection
}

def convert_xml_to_yolo(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        label = obj.find("name").text.strip()
        if label not in CLASS_MAP:
            print(f"  WARNING: unknown label '{label}' — skipping")
            continue

        class_id = CLASS_MAP[label]
        bndbox   = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return yolo_lines


def find_image(base_name):
    """Try all common image extensions — returns (full_path, filename) or None."""
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
        candidate = os.path.join(IMAGES_DIR, base_name + ext)
        if os.path.exists(candidate):
            return candidate, base_name + ext
    return None, None


def main():
    for split in ["train", "val"]:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}",  exist_ok=True)

    xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".xml")]
    print(f"Total XML files found: {len(xml_files)}")

    pairs = []
    for xml_name in xml_files:
        base_name = xml_name.replace(".xml", "")
        xml_path  = os.path.join(ANNOTATIONS_DIR, xml_name)
        img_path, img_name = find_image(base_name)

        if img_path:
            pairs.append((img_path, xml_path, img_name, xml_name))
        else:
            print(f"  MISSING image for {xml_name} — skipping")

    print(f"Matched {len(pairs)} image-annotation pairs")

    if len(pairs) == 0:
        print("\nERROR: No pairs found!")
        print("Images in your folder:")
        for f in os.listdir(IMAGES_DIR)[:10]:
            print(f"  {f}")
        return

    random.seed(42)
    random.shuffle(pairs)
    split_idx   = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    val_pairs   = pairs[split_idx:]

    print(f"Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    skipped = 0
    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, xml_path, img_name, xml_name in split_pairs:
            yolo_lines = convert_xml_to_yolo(xml_path)

            if not yolo_lines:
                skipped += 1
                continue

            shutil.copy(img_path, f"{OUTPUT_DIR}/images/{split}/{img_name}")

            label_name = os.path.splitext(img_name)[0] + ".txt"
            with open(f"{OUTPUT_DIR}/labels/{split}/{label_name}", "w") as f:
                f.write("\n".join(yolo_lines))

    yaml_content = f"""path: ./{OUTPUT_DIR}
train: images/train
val:   images/val

nc: 2
names:
  - helmet
  - no_helmet
"""
    with open(f"{OUTPUT_DIR}/helmet.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\nDone!")
    print(f"  Skipped (no valid labels): {skipped}")
    print(f"  Dataset ready in: {OUTPUT_DIR}/")
    print(f"\nNext: python yolov5/train.py --data {OUTPUT_DIR}/helmet.yaml --weights yolov5s.pt --epochs 50 --img 640")

if __name__ == "__main__":
    main()