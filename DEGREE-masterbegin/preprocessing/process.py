import json

# Read the JSON string list file containing escape characters
with open('output_data_prompt.json', 'r', encoding='utf-8') as file:
    raw_data = json.load(file)

# Process the data, converting strings to dictionary objects
processed_data = []
for item in raw_data:
    # Use the split() method to split the string, ensuring each part is a valid JSON string
    raw_items = item.split('},{')
    for i in range(len(raw_items)):
        if not raw_items[i].startswith('{'):
            raw_items[i] = '{' + raw_items[i]
        if not raw_items[i].endswith('}'):
            raw_items[i] = raw_items[i] + '}'

    # Parse each valid JSON string into a dictionary object
    for raw_item in raw_items:
        try:
            json_item = json.loads(raw_item)
            # Replace double quotes with single quotes in the sentence
            if "sentence" in json_item:
                json_item["sentence"] = json_item["sentence"].replace('"', "'")
            processed_data.append(json_item)
        except json.JSONDecodeError as e:
            print(f"Problematic item: {raw_item}")

# Output to JSON file
with open('processed_data/data_LLM/1-shotchat.json', 'w', encoding='utf-8') as output_file:
    json.dump(processed_data, output_file, ensure_ascii=False, indent=4)

print("Data has been successfully processed and saved to the 1-shotchat.json file.")
