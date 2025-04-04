import os
import xml.etree.ElementTree as ET
import json

base_path = 'datasets/SysPhs-Physical-Interaction-Libraries-and-Manufacturing-Examples-release'

extracted_data = []

def extract_names_from_xml(elem, path_prefix=None, file_name=None):
    if path_prefix is None:
        path_prefix = []

    tag = elem.tag.split('}')[-1]  # Strip namespace if present
    current_path = path_prefix + [tag]

    name = elem.attrib.get('name')
    if name and name.strip():
        record = {
            'name': name.strip(),
            'path': current_path,
            'dataset': 'SysPhs',
            'model-name': file_name
        }
        for attr in ['id']:  # Keeping bpmnElement in case reused elements exist
            if attr in elem.attrib:
                record[attr] = elem.attrib[attr].strip()
        extracted_data.append(record)

    for child in elem:
        extract_names_from_xml(child, current_path, file_name)

# Process all .xml files in the base path
for file in os.listdir(base_path):
    if file.endswith('.xml'):
        file_path = os.path.join(base_path, file)
        file_name = os.path.splitext(file)[0]

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            extract_names_from_xml(root, file_name=file_name)
        except ET.ParseError:
            print(f"❌ Could not parse XML: {file_path}")

# Save output
with open('extracted_sysphs.json', 'w', encoding='utf-8') as out_file:
    json.dump(extracted_data, out_file, indent=2, ensure_ascii=False)

print(f"✅ Extracted {len(extracted_data)} name entries with paths and metadata to extracted_sysphs.json")
