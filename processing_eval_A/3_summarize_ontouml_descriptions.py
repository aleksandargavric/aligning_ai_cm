import csv
import ollama
import json
import sys

# Filenames for input and output
input_filename = 'ontouml_models_serializations.csv'
output_filename = 'ontouml_models_descriptions.csv'

# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)

# Relationship Extraction Prompt Template
prompt_template = """
    OntoUML model: {serialization}
    
    Task: Analyze the OntoUML model provided in the serialization and generate a concise description that outlines **how elements are connected**. 

    Instructions:
    - Go through all internal element identifiers (IDs) and identify the relationships between them.
    - For each connection, preserve and use the natural-language names of the involved elements.
    - Summarize the connections between classifiers (e.g., classes, relators, roles) and describe their nature (e.g., mediations, generalizations, associations).
    - Do not describe unrelated attributes or metadata.
    
    Output:
    - A structured description of the model's connectivity, focusing only on inter-element relationships.
    - Use OntoUML-relevant terminology (e.g., "relator mediates between", "role played by", "generalization from X to Y").
    - Avoid listing raw IDs, but ensure every named element mentioned is traceable through its name.

    The output should serve as a compact summary of the model's structure through its relationships.
"""


# Open the input CSV for reading and output CSV for writing
with open(input_filename, newline='', encoding='utf-8') as infile, \
     open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=['key', 'description'])
    writer.writeheader()
    
    # Process each row from the input CSV
    for row in reader:
        serialization = row['serialization']
        prompt = prompt_template.format(serialization=serialization)
        
        # Call the Ollama API with the generated prompt
        result = ollama.generate(model='llama3.3', prompt=prompt)
        description = result.get('response', '').strip()
        print(description)
        
        # Write the key and corresponding description to the output CSV
        writer.writerow({'key': row['key'], 'description': description})

print(f"Descriptions saved to {output_filename}")