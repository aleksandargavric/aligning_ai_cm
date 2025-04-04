import os
import xml.etree.ElementTree as ET
import json

base_path = 'datasets/hdBPMN-main/data/annotations'

extracted_data = []

def extract_names_from_xml(elem, path_prefix=None, folder_name=None):
    if path_prefix is None:
        path_prefix = []

    tag = elem.tag.split('}')[-1]  # Remove namespace
    current_path = path_prefix + [tag]

    name = elem.attrib.get('name')
    if name and name.strip():
        record = {
            'name': name.strip(),
            'path': current_path,
            'dataset': 'modelset',
            'model-name': folder_name
        }
        for attr in ['bpmnElement', 'id']:
            if attr in elem.attrib:
                record[attr] = elem.attrib[attr].strip()
        extracted_data.append(record)

    for child in elem:
        extract_names_from_xml(child, current_path, folder_name)

# Walk through all folders and files
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.endswith('.bpmn'):
            file_path = os.path.join(folder_path, file)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                extract_names_from_xml(root, folder_name=folder)
            except ET.ParseError:
                print(f"❌ Could not parse XML: {file_path}")

# Save output
with open('extracted_hdbpmn.json', 'w', encoding='utf-8') as out_file:
    json.dump(extracted_data, out_file, indent=2, ensure_ascii=False)

print(f"✅ Extracted {len(extracted_data)} name entries with paths and metadata to extracted_hdbpmn.json")
