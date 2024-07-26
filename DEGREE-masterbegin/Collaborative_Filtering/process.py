import json

'''
From:
{"doc_id": "CNN_CF_20030303.1900.02", "wnd_id": "CNN_CF_20030303.1900.02-0", "entity_mentions": [{"id": "CNN_CF_20030303.1900.02-0-E0", "start": 0, "end": 2, "entity_type": "PER", "mention_type": "UNK", "text": "Wouter Basson"}, {"id": "CNN_CF_20030303.1900.02-0-E1", "start": 0, "end": 2, "entity_type": "PER", "mention_type": "UNK", "text": "Wouter Basson"}], "relation_mentions": [], "event_mentions": [{"event_type": "Justice:Acquit", "id": "CNN_CF_20030303.1900.02-0-EV0", "trigger": {"start": 3, "end": 4, "text": "acquitted"}, "arguments": [{"entity_id": "CNN_CF_20030303.1900.02-0-E1", "text": "Wouter Basson", "role": "Defendant"}]}, {"event_type": "Justice:Charge-Indict", "id": "CNN_CF_20030303.1900.02-0-EV1", "trigger": {"start": 9, "end": 10, "text": "charges"}, "arguments": [{"entity_id": "CNN_CF_20030303.1900.02-0-E1", "text": "Wouter Basson", "role": "Defendant"}]}], "entity_coreference": [], "event_coreference": [], "tokens": ["Wouter", "Basson", "was", "acquitted", "in", "April", "2002", "on", "46", "charges", ",", "ranging", "from", "murder", "and", "drug", "trafficking", "to", "fraud", "and", "theft", "."], "pieces": ["W", "outer", "B", "ass", "on", "was", "acqu", "itted", "in", "April", "2002", "on", "46", "charges", ",", "ranging", "from", "mur", "der", "and", "drug", "tra", "ff", "icking", "to", "f", "raud", "and", "the", "ft", "."], "token_lens": [2, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 2, 1, 2, 1], "sentence": "Wouter Basson was acquitted in April 2002 on 46 charges , ranging from murder and drug trafficking to fraud and theft .", "sentence_starts": [0]}
Convert to:
{"sentence": "on wednesday , washington wizards ' owner ended the three - year association with jordan .", "events_info":[
    {"trigger_text": "ended", "event_type": "Personnel:End-Position", "arguments": [
        {"text": "jordan", "role": "Person"}, 
        {"text": "washington wizards", "role": "Entity"}
    ]}
]}
'''

# Read JSON file and process it line by line
with open('data1.json', 'r') as file:
    data1 = json.load(file)

sentences = [data1["sentence"] for data1 in data1]
events_info_all = []
with open('DEGREE-masterbegin/processed_data/ACE05ep_bart/train.w1.oneie.json', 'r') as file:
    for line in file:
        data = json.loads(line)
        # print(data["sentence"])
        if data["sentence"] in sentences:
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
