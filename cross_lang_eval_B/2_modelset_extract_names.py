import os
import json

base_path = 'datasets/modelset/graph/repo-genmymodel-uml/data/'

extracted_data = []

def read_json_with_fallback(path):
    encodings_to_try = ['utf-8', 'latin-1', 'windows-1252']
    for enc in encodings_to_try:
        try:
            with open(path, 'r', encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    print(f"❌ Could not decode or parse: {path}")
    return None

def extract_names(obj, path_prefix=None, folder_name=None):
    if path_prefix is None:
        path_prefix = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = path_prefix + [key]
            if key == 'name' and isinstance(value, str) and value.strip():
                record = {
                    'name': value.strip(),
                    'path': current_path,
                    'dataset': 'modelset',
                    'model-name': folder_name
                }
                # Try to fetch attributes from the same object
                for attr in ['eClass', 'qualifiedName']:
                    if attr in obj and isinstance(obj[attr], str) and obj[attr].strip():
                        record[attr] = obj[attr].strip()
                extracted_data.append(record)
            else:
                extract_names(value, current_path, folder_name)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            extract_names(item, path_prefix + [f"[{index}]"], folder_name)

# Walk through all folders
for folder in os.listdir(base_path):
    ontology_path = os.path.join(base_path, folder, folder[:-4] + '.json')
    if os.path.isfile(ontology_path):
        data = read_json_with_fallback(ontology_path)
        if data:
            extract_names(data, folder_name=folder)

# Save output
with open('extracted_modelset.json', 'w', encoding='utf-8') as out_file:
    json.dump(extracted_data, out_file, indent=2, ensure_ascii=False)

print(f"✅ Extracted {len(extracted_data)} name entries with paths and metadata to extracted_modelset.json")
