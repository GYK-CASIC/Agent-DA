# Constructing a Ranking Dataset
# Input: Original low-resource data
# Output: Ranking dataset
import random
import json
# python processcopy.py

# Read input data file
input_file_LLM = 'processed_data/data_LLM/1-shot.json'
with open(input_file_LLM, 'r') as file:
    input_data_LLM = json.load(file)

import random
event_types = ["Movement:Transport","Personnel:Elect","Personnel:Start-Position","Personnel:Nominate",
    "Personnel:End-Position",
    "Conflict:Attack",
    "Contact:Meet",
    "Life:Marry",
    "Transaction:Transfer-Money",
    "Conflict:Demonstrate",
    "Business:End-Org",
    "Justice:Sue",
    "Life:Injure",
    "Life:Die",
    "Justice:Arrest-Jail",
    "Contact:Phone-Write",
    "Transaction:Transfer-Ownership",
    "Business:Start-Org",
    "Justice:Execute",
    "Justice:Trial-Hearing",
    "Life:Be-Born",
    "Justice:Charge-Indict",
    "Justice:Convict",
    "Justice:Sentence",
    "Business:Declare-Bankruptcy",
    "Justice:Release-Parole",
    "Justice:Fine",
    "Justice:Pardon",
    "Justice:Appeal",
    "Justice:Extradite",
    "Life:Divorce",
    "Business:Merge-Org",
    "Justice:Acquit"
]

argument_role = ["Origin", "Entity", "Agent", "Destination", "Person", "Artifact", "Vehicle", "Target", "Attacker", "Victim", "Instrument", "Plaintiff", "Adjudicator", "Defendant", "Prosecutor", "Seller", "Buyer", "Beneficiary", "Giver", "Org"]
ranked_data = []
ranked_data2 = []
ranked_data3 = []
ranked_data4 = []
# Complete event annotations
for item in input_data_LLM:
    for _ in range(5):
        sentence = item['sentence']
        events_info = item['events_info']
        event_strings = []
        # Iterate through each event info
        for event_info in events_info:
            event_string = f"trigger_text is {event_info['trigger_text']}, event_type is {event_info['event_type']},"
            if event_info['arguments']:
                event_string += ','.join([f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in event_info['arguments']])
            event_strings.append(event_string)

            # Combine all event types into the same row
        complete_event_string = '；'.join(event_strings)
        ranked_data.append(f"{sentence} {complete_event_string}")
        # print(ranked_data)

# Event annotations with incorrect argument information
import random

# Event annotations with incorrect argument information
for item in input_data_LLM:
    sentence = item['sentence']
    events_info = item['events_info']
    event_strings2 = []
    # Iterate through each event info
    for event_info in events_info:
        event_string2 = f"trigger_text is {event_info['trigger_text']}, event_type is {event_info['event_type']},"
        if event_info['arguments']:
            new_arguments = []
            for arg in event_info['arguments']:
                # Randomly decide whether to replace the current argument
                if random.random() < 0.5:  # 50% chance to replace
                    new_text = random.choice(sentence.split())
                    new_role = "wrong_role"  # Incorrect role
                    new_arguments.append(f"text: \"{new_text}\", role: \"{arg['role']}\"")
                else:
                    new_arguments.append(f"text: \"{arg['text']}\", role: \"{arg['role']}\"")
            event_string2 += ','.join(new_arguments)
        # Add processed event info to the list
        event_strings2.append(event_string2)


    # Combine all event types into the same row
    complete_event_string2 = '；'.join(event_strings2)
    ranked_data2.append(f"{sentence} {complete_event_string2}")
# Randomly delete argument information
for item in input_data_LLM:
    sentence = item['sentence']
    events_info = item['events_info']
    event_strings3 = []
    # Iterate through each event info
    for event_info in events_info:
        event_string3 = f"trigger_text is {event_info['trigger_text']}, event_type is {event_info['event_type']},"
        if event_info['arguments']:
            # Event annotations with missing argument information
            arguments = ",".join([f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in event_info['arguments']])
            # Number of arguments
            num_arguments = len(event_info['arguments'])
            # print(num_arguments)
            # Randomly delete one or more arguments
            # print(num_arguments)
            if num_arguments > 0:
                num_deleted_arguments = random.randint(0, num_arguments - 1)
            else:
                num_deleted_arguments = 0
            # print(num_deleted_arguments)
            deleted_arguments = random.sample(event_info['arguments'], num_deleted_arguments)
            event_string3 += ','.join([f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in deleted_arguments])
        event_strings3.append(event_string3)

    # Combine all event types into the same row
    complete_event_string3 = '；'.join(event_strings3)
    ranked_data3.append(f"{sentence} {complete_event_string3}")
# Randomly add argument information
for item in input_data_LLM:
    sentence = item['sentence']
    events_info = item['events_info']
    event_strings4 = []
    # Iterate through each event info
    for event_info in events_info:
        event_string4 = f"trigger_text is {event_info['trigger_text']}, event_type is {event_info['event_type']},"
        if event_info['arguments']:
            num_new_arguments = random.randint(1,5)
            new_arguments = []
            for _ in range(num_new_arguments):
                new_argument = random.choice(sentence.split())
                new_role = random.choice(argument_role)
                new_arguments.append((new_argument,new_role))
            existing_arguments = ",".join([f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in event_info['arguments']])
            new_arguments = ",".join([f"text: \"{arg}\", role: \"{role}\"" for arg, role in new_arguments])
            combined_arguments = "".join(existing_arguments + new_arguments)

            event_string4 += "".join(combined_arguments)
        event_strings4.append(event_string4)

    # Combine all event types into the same row
    complete_event_string4 = ','.join(event_strings4)
    ranked_data4.append(f"{sentence} {complete_event_string4}")

# Event annotations with incorrect trigger word
ranked_data5 = []
for item in input_data_LLM:
    for _ in range(1):
        sentence = item['sentence']
        events_info = item['events_info']
        random_trigger = random.choice(sentence.split())
        event_strings = []
        # Iterate through each event info
        for event_info in events_info:
            event_string = f"trigger_text is {random_trigger}, event_type is {event_info['event_type']},"
            if event_info['arguments']:
                event_string += ','.join([f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in event_info['arguments']])
            event_strings.append(event_string)

            # Combine all event types into the same row
        complete_event_string = '；'.join(event_strings)
        ranked_data5.append(f"{sentence} {complete_event_string}")
# Event annotations with incorrect event type
ranked_data6 = []
for item in input_data_LLM:
    for _ in range(1):
        sentence = item['sentence']
        events_info = item['events_info']
        random_trigger = random.choice(sentence.split())
        event_strings = []
        # Iterate through each event info
        for event_info in events_info:
            events = event_types.remove(event_info['event_type'])
            # print(event_types)
            random_event_type = random.choice(event_types)
            event_types.append(event_info['event_type'])
            event_string = f"trigger_text is {event_info['trigger_text']}, event_type is {random_event_type},"
            if event_info['arguments']:
                event_string += ','.join([f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in event_info['arguments']])
            event_strings.append(event_string)

            # Combine all event types into the same row
        complete_event_string = '；'.join(event_strings)
        ranked_data6.append(f"{sentence} {complete_event_string}")
# Save the ranked dataset as a TSV file
ranked_data_all = []
ranked_data_final = []
for item1,item2,item3,item4,item5 in zip(ranked_data2,ranked_data3,ranked_data4,ranked_data5,ranked_data6):
    ranked_data_all.append(item1)
    ranked_data_all.append(item2)
    ranked_data_all.append(item3)
    ranked_data_all.append(item4)
    ranked_data_all.append(item5)

    # print(ranked_data_all)
for item1,item_all in zip(ranked_data,ranked_data_all):
    ranked_data_final.append(item1)
    ranked_data_final.append(item_all)

output_file = 'RLHF/data/reward_datasets/sentiment_analysis/train.tsv'
with open(output_file, 'w') as file:
    count = 0
    for item in ranked_data_final:
        if count % 2 == 0 and count != 0:
            file.write('\n')
        file.write(f"{item}\t")
        count += 1

print(f"Ranked dataset has been saved to {output_file}")
