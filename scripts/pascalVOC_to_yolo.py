import os
import xml.etree.ElementTree as ET
import argparse

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, txt_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(txt_file, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")

def main(input_dir, output_dir):
    classes = open(os.path.join(input_dir, 'classes.txt')).read().strip().split()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    for xml_file in xml_files:
        txt_file = os.path.join(output_dir, os.path.splitext(xml_file)[0] + '.txt')
        convert_annotation(os.path.join(input_dir, xml_file), txt_file, classes)
        print(f"Converted {xml_file} to {txt_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVOC annotations to YOLOv8 format.")
    parser.add_argument("--input_dir", type=str, help="Directory containing the PascalVOC XML files.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the YOLOv8 TXT files.")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
