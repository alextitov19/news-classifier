# convert_json.py
import json

def convert_to_json_array(input_file, output_file):
    data = []
    with open(input_file, 'r') as infile:
        for line in infile:
            if line.strip():  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping line due to JSONDecodeError: {e}")

    with open(output_file, 'w') as outfile:
        json.dump({"root": data}, outfile, indent=4)

if __name__ == '__main__':
    convert_to_json_array('data/articles.json', 'data/articles_valid.json')