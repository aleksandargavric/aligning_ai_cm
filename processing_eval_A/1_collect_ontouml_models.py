import os
import csv
import yaml

# Directory containing the model folders
models_dir = os.path.join('..', 'datasets', 'ontouml-models-master', 'models')

# Output CSV file path
output_csv = 'ontouml_models.csv'

# CSV header with the desired columns
header = [
    'key',
    'title',
    'keywords',
    'theme',
    'ontologyType',
    'designedForTask',
    'language',
    'context',
    'source'
]

rows = []

# Iterate over each folder in the models directory
for folder in os.listdir(models_dir):
    folder_path = os.path.join(models_dir, folder)
    if os.path.isdir(folder_path):
        metadata_path = os.path.join(folder_path, 'metadata.yaml')
        if os.path.isfile(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                try:
                    data = yaml.safe_load(f)
                except Exception as e:
                    print(f"Error reading YAML file {metadata_path}: {e}")
                    data = {}
            
            # Retrieve each value; if the key doesn't exist, default to an empty string
            title = data.get('title', '')
            keywords = data.get('keywords', '')
            # Convert keywords to a string if it's a list
            if isinstance(keywords, list):
                keywords = ', '.join(keywords)
            else:
                keywords = str(keywords)
            theme = data.get('theme', '')
            ontologyType = data.get('ontologyType', '')
            designedForTask = data.get('designedForTask', '')
            language = data.get('language', '')
            context = data.get('context', '')
            source = data.get('source', '')
            
            rows.append([
                folder,
                title,
                keywords,
                theme,
                ontologyType,
                designedForTask,
                language,
                context,
                source
            ])
        else:
            print(f"metadata.yaml not found in {folder_path}, skipping folder.")

# Write all collected rows to the CSV file
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(rows)

print(f"CSV file '{output_csv}' has been created with {len(rows)} records.")
