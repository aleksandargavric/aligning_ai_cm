import os
import xml.etree.ElementTree as ET
import json

base_path = 'datasets/bpmn-for-research-master/BPMN for Research/English'

extracted_data = []

def extract_names_from_xml(elem, path_prefix=None, folder_name=None):
    if path_prefix is None:
        path_prefix = []

    tag = elem.tag.split('}')[-1]  # Remove namespace if present
    current_path = path_prefix + [tag]

    name = elem.attrib.get('name')
    if name and name.strip():
        record = {
            'name': name.strip(),
            'path': current_path,
            'dataset': 'bpmnResearch',
            'model-name': folder_name
        }
        for attr in ['bpmnElement', 'id']:
            if attr in elem.attrib:
                record[attr] = elem.attrib[attr].strip()
        extracted_data.append(record)

    for child in elem:
        extract_names_from_xml(child, current_path, folder_name)

# Walk through all nested folders to find .bpmn files
for root, _, files in os.walk(base_path):
    for file in files:
        if file.endswith('.bpmn'):
            file_path = os.path.join(root, file)
            model_name = os.path.basename(root)  # Use immediate folder as model-name

            try:
                tree = ET.parse(file_path)
                root_elem = tree.getroot()
                extract_names_from_xml(root_elem, folder_name=model_name)
            except ET.ParseError:
                print(f"❌ Could not parse XML: {file_path}")

# Save output
with open('extracted_bpmnresearch.json', 'w', encoding='utf-8') as out_file:
    json.dump(extracted_data, out_file, indent=2, ensure_ascii=False)

print(f"✅ Extracted {len(extracted_data)} name entries with paths and metadata to extracted_bpmnresearch.json")
