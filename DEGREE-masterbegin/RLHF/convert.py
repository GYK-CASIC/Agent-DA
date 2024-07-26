import json

# Read the JSON file and process it line by line
events_info_all = []
with open('test.json', 'r') as file:
    for line in file:
        data = json.loads(line)

        # Extract event information
        events_info = []
        for event in data['event_mentions']:
            trigger_text = event['trigger']['text']
            event_type = event['event_type']
            arguments = []
            for arg in event['arguments']:
                arguments.append({'text': arg['text'], 'role': arg['role']})
            events_info.append({'trigger_text': trigger_text, 'event_type': event_type, 'arguments': arguments})

            converted_data = {'sentence': data['sentence'], 'events_info': events_info}

            events_info_all.append(converted_data)
# Construct the converted data format
converted_data = {'events_info_all': events_info_all}

# Save the converted data as a JSON file
with open('converted_data.json', 'w') as output_file:
    json.dump(converted_data, output_file, indent=4)

print("Conversion complete!")
