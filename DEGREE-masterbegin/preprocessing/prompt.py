import json
import tiktoken
from openai import OpenAI

# Initialize OpenAI client
api_key = ''
client = OpenAI(api_key=api_key)

# Read input data
with open('processed_data/data_LLM/1-shot.json', 'r', encoding='utf-8') as file:
    input_data = json.load(file)

# Define function to calculate number of tokens
def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += -1
        num_tokens += 2
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

# Define request function
def generate_augmented_data(sentence, events_info):
    system_message = {
        "role": "system",
        "content": "You are a sample generator. Generate #5# sentences that contain the same 'trigger_text' and 'event_type' as the input sentence. Each generated sentence should maintain the correct argument roles specified for the event_type. Try to include all argument roles in the output. \"event_schemas\": { \"Movement:Transport\": [\"Vehicle\", \"Artifact\", \"Agent\", \"Origin\", \"Destination\"], \"Personnel:Elect\": [\"Place\", \"Person\", \"Entity\"], \"Personnel:Start-Position\": [\"Place\", \"Person\", \"Entity\"], \"Personnel:Nominate\": [\"Agent\", \"Person\"], \"Personnel:End-Position\": [\"Place\", \"Person\", \"Entity\"], \"Conflict:Attack\": [\"Target\", \"Place\", \"Victim\", \"Instrument\", \"Attacker\"], \"Contact:Meet\": [\"Place\", \"Entity\"], \"Life:Marry\": [\"Place\", \"Person\"], \"Transaction:Transfer-Money\": [\"Giver\", \"Place\", \"Recipient\", \"Beneficiary\"], \"Conflict:Demonstrate\": [\"Place\", \"Entity\"], \"Business:End-Org\": [\"Place\", \"Org\"], \"Justice:Sue\": [\"Defendant\", \"Plaintiff\", \"Adjudicator\", \"Place\"], \"Life:Injure\": [\"Agent\", \"Place\", \"Victim\", \"Instrument\"], \"Life:Die\": [\"Person\", \"Agent\", \"Place\", \"Victim\", \"Instrument\"], \"Justice:Arrest-Jail\": [\"Agent\", \"Place\", \"Person\"], \"Contact:Phone-Write\": [\"Place\", \"Entity\"], \"Transaction:Transfer-Ownership\": [\"Artifact\", \"Beneficiary\", \"Buyer\", \"Place\", \"Seller\"], \"Business:Start-Org\": [\"Agent\", \"Place\", \"Org\"], \"Justice:Execute\": [\"Agent\", \"Place\", \"Person\"], \"Justice:Trial-Hearing\": [\"Prosecutor\", \"Defendant\", \"Place\", \"Adjudicator\"], \"Life:Be-Born\": [\"Place\", \"Person\"], \"Justice:Charge-Indict\": [\"Prosecutor\", \"Adjudicator\", \"Place\", \"Defendant\"], \"Justice:Convict\": [\"Defendant\", \"Place\", \"Adjudicator\"], \"Justice:Sentence\": [\"Adjudicator\", \"Place\", \"Defendant\"], \"Business:Declare-Bankruptcy\": [\"Place\", \"Org\"], \"Justice:Release-Parole\": [\"Place\", \"Person\", \"Entity\"], \"Justice:Fine\": [\"Adjudicator\", \"Place\", \"Entity\"], \"Justice:Pardon\": [\"Adjudicator\", \"Place\", \"Defendant\"], \"Justice:Appeal\": [\"Adjudicator\", \"Plaintiff\", \"Place\"], \"Justice:Extradite\": [\"Agent\", \"Origin\", \"Destination\"], \"Life:Divorce\": [\"Place\", \"Person\"], \"Business:Merge-Org\": [\"Org\"], \"Justice:Acquit\": [\"Defendant\", \"Adjudicator\"] },\n Please separate each data entry with a \",\"\n 请你按照输出格式进行输出."
    }
    # Generate a list of event details including event type and trigger word for each event
    events_details = [
        {
            "event_type": event["event_type"],
            "trigger_text": event["trigger_text"],
            "arguments": event["arguments"]
        }
        for event in events_info
    ]

    # List each event and its corresponding trigger word in user_message
    user_message_content = (f"\"sentence\": \"{sentence}\", \"events_info\": {json.dumps(events_info)}\nBased on the input, please generate 5 new samples.")
    user_message = {
        "role": "user",
        "content": user_message_content
    }
    example_messages = [
        {
            "role": "user",
            "content": "{\"sentence\": \"When Franklin Roosevelt was elected president, the south still felt like a conquered nation after the Civil War.\",\"events_info\": [{\"trigger_text\": \"elected\", \"event_type\": \"Personnel:Elect\", \"arguments\": [{\"text\": \"Franklin Roosevelt\",\"role\": \"Person\"}, {\"text\": \"president\", \"role\": \"Entity\"}]}]}\nBased on the input, generate five new samples."
        },
        {
            "role": "assistant",
            "content": "{\"sentence\": \"When Franklin Roosevelt was elected president, the country felt a sense of hope and change.\", \"events_info\": [{\"trigger_text\": \"elected\", \"event_type\": \"Personnel:Elect\", \"arguments\": [{\"text\": \"Franklin Roosevelt\",\"role\": \"Person\"}, {\"text\": \"president\", \"role\": \"Entity\"}]}]}"
        }
    ]

    messages = [system_message] + example_messages + [user_message]

    # Calculate input tokens
    input_tokens = num_tokens_from_messages(messages)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500,
            n=1,
            stop=None,
            temperature=0.7,
            top_p=20
        )
        response_message = response.choices[0].message.content
        return response_message
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        with open('output.txt', 'w') as f:
            f.write(user_message['content'])
        return None

output_data = []
augmented_data_list = []

for entry in input_data:
    sentence = entry["sentence"]
    events_info = entry["events_info"]
    print(entry["sentence"])
    augmented_data = generate_augmented_data(sentence, events_info)
    print(augmented_data)
    if augmented_data:
        augmented_data_list.append(augmented_data)
    else:
        print(entry)

with open('output_data_prompt.json', 'w', encoding='utf-8') as output_file:
    json.dump(augmented_data_list, output_file, ensure_ascii=False, indent=4)
