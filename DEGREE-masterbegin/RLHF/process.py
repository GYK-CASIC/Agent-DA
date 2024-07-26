import random
import json

# Read input data file
# python RLHF/process.py
input_file_SLM = 'Collaborative_Filtering/diff_SLM.json'
with open(input_file_SLM, 'r') as file:
    input_data_SLM = json.load(file)

input_file_LLM = 'Collaborative_Filtering/diff_LLM.json'
with open(input_file_LLM, 'r') as file:
    input_data_LLM = json.load(file)

event_types = [
    "Movement:Transport",
    "Personnel:Elect",
    "Personnel:Start-Position",
    "Personnel:Nominate",
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

argument_role = ["Origin", "Entity", "Agent", "Destination", "Person", "Artifact", "Vehicle", "Target", "Attacker",
                 "Victim", "Instrument", "Plaintiff", "Adjudicator", "Defendant", "Prosecutor", "Seller", "Buyer",
                 "Beneficiary", "Giver", "Org"]

# Generate ranked dataset

data1 = []
data2 = []
ranked_data = []
for item in input_data_LLM:
    sentence = item['sentence']
    events_info = item['events_info']

    # Event annotation with complete information
    complete_event = {
        'sentence': sentence,
        'trigger_text': events_info[0]['trigger_text'],
        'event_type': events_info[0]['event_type'],
        'arguments': ','.join(
            [f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in events_info[0]['arguments']])
    }
    data1.append(complete_event)

for item in input_data_SLM:
    print(item)
    sentence = item['sentence']
    events_info = item['events_info']
    # Event annotation with complete information
    complete_event = {
        'sentence': sentence,
        'trigger_text': events_info[0]['trigger_text'],
        'event_type': events_info[0]['event_type'],
        'arguments': ','.join(
            [f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in events_info[0]['arguments']])
    }
    print(complete_event)
    data2.append(complete_event)
# print(data1)

for item1, item2 in zip(data1, data2):
    ranked_data.append(item1)
    ranked_data.append(item2)

print(ranked_data)
'''
input_file_LLM = 'test.json'
with open(input_file_LLM, 'r') as file:
    input_data_LLM = json.load(file)

event_types = [
    "Movement:Transport",
    "Personnel:Elect",
    "Personnel:Start-Position",
    "Personnel:Nominate",
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
for item in input_data_LLM:
    sentence = item['sentence']
    events_info = item['events_info']
    print(len(events_info))
    print(events_info[0])
    for _ in range(4):
        # Event annotation with complete information
        # Check the number of event types
        complete_event = {
        'sentence': sentence,
        'trigger_text': events_info[0]['trigger_text'],
        'event_type': events_info[0]['event_type'],
        'arguments': ','.join([f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in events_info[0]['arguments']])}

        ranked_data.append(complete_event)

        # Event annotation with incorrect argument
        wrong_argument_event = {
        'sentence': sentence,
        'trigger_text': events_info[0]['trigger_text'],
        'event_type': events_info[0]['event_type'],
        'arguments': ','.join([f"text: \"{random.choice(sentence.split())}\", role: \"{arg['role']}\"" for arg in events_info[0]['arguments']])
    }

        # Event annotation with missing argument information
        arguments = ",".join([f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in events_info[0]['arguments']])
        # Number of arguments
        num_arguments = len(events_info[0]['arguments'])
        # print(num_arguments)
        # Randomly delete one or more arguments
        # print(num_arguments)
        if num_arguments > 0:
            num_deleted_arguments = random.randint(0, num_arguments - 1)
        else:
            num_deleted_arguments = 0
        # print(num_deleted_arguments)
        deleted_arguments = random.sample(events_info[0]['arguments'], num_deleted_arguments)
        missing_argument_event = {
        'sentence': sentence,
        'trigger_text': events_info[0]['trigger_text'],
        'event_type': events_info[0]['event_type'],
        'arguments': arguments.replace(','.join([f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in deleted_arguments]), '')
        }

        # Event annotation with randomly added redundant information
        num_new_arguments = random.randint(1, 5)
        new_arguments = []
        for _ in range(num_new_arguments):
            new_argument = random.choice(sentence.split())
            new_role = random.choice(argument_role)
            new_arguments.append((new_argument, new_role))
        existing_arguments = ",".join([f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in events_info[0]['arguments']])
        new_arguments = ",".join([f"text: \"{arg}\", role: \"{role}\"" for arg, role in new_arguments])
        combined_arguments = "".join(existing_arguments + new_arguments)

        redundant_event = {
        'sentence': sentence,
        'trigger_text': events_info[0]['trigger_text'],
        'event_type': events_info[0]['event_type'],
        'arguments': f"{combined_arguments}"
    }
        argument_choice = [wrong_argument_event, missing_argument_event, redundant_event]
        ranked_data.append(random.choice(argument_choice))

        # Event annotation with incorrect trigger word
        random_trigger = random.choice(sentence.split())  # Randomly choose a word from the sentence as the trigger word

        wrong_trigger_event = {
        'sentence': sentence,
        'trigger_text': random_trigger,
        'event_type': events_info[0]['event_type'],
        'arguments': ','.join([f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in events_info[0]['arguments']])
    }
        # ranked_data.append(wrong_trigger_event)

        # Event annotation with incorrect event type
        wrong_type_event = {
        'sentence': sentence,
        'trigger_text': events_info[0]['trigger_text'],
        'event_type': random.choice(event_types),  # Randomly choose from the given options
        'arguments': ','.join([f"text: \"{arg['text']}\", role: \"{arg['role']}\"" for arg in events_info[0]['arguments']])
        }
        #ranked_data.append(wrong_type_event)

        choice = [wrong_argument_event, missing_argument_event, redundant_event, wrong_trigger_event, wrong_type_event]
        ranked_data.append(random.choice(choice))
'''
# Save the ranked dataset as a TSV file, writing four sentences per line
output_file = 'RLHF/data/reward_datasets/sentiment_analysis/test.tsv'
with open(output_file, 'w') as file:
    count = 0
    for item in ranked_data:
        if count % 2 == 0 and count != 0:
            file.write('\n')
        file.write(
            f"{item['sentence']} trigger_text is {item['trigger_text']}, event_type is {item['event_type']}, {item['arguments']}\t")
        count += 1

print(f"Ranked dataset has been saved to {output_file}")
