import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load Mixtral Model (Following `51-mixtral_solved_predictor_questions_only.py`)

mistral_checkpoint = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load the tokenizer
print(f"[INFO] Loading Mistral model from: {mistral_checkpoint}")
tokenizer = AutoTokenizer.from_pretrained(mistral_checkpoint)
tokenizer.pad_token = tokenizer.eos_token  # Ensure correct padding behavior

# Enable 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model with quantization and optimized device mapping
model = AutoModelForCausalLM.from_pretrained(mistral_checkpoint, quantization_config=bnb_config, device_map="auto")
model.eval()

# File Paths
INPUT_FILE = "data/processed/preprocessed_dataset.json"
OUTPUT_FILE = "data/processed/labeled_dataset.json"

# Load Preprocessed Data
def load_data(input_path):
    """Loads the preprocessed conversation dataset."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Error: {input_path} not found!")
    
    return pd.read_json(input_path, lines=True)

# Define Mixtral Query Function

def query_mixtral(prompt, max_tokens=512):
    """
    Sends a prompt to the Mixtral model and returns the generated response.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True
        )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text

# Entity Extraction
def extract_entities(message):

    prompt = f"""As an expert in Named Entity Recognition (NER) specialized in analyzing chatrooms discussions about computer programming concepts, your task is to identify tokens from the given block of messages of a question in s conversation that correspond to a provided list of computer programming-related delimited by *** {entities}. 
    The values associated with the "text" keys are the content to be analyzed for entity extraction. If a token in the value associated with the "text" is an entity then you must extract it.
    For each identified token that matches one of the provided entities, report the output in the following format: 
    [ENTITY]: [ "timestamp": "XXX", "token": "XXXXX"]. 
    In the output, the timestamp is associated with the message in which the token appears, which can be found in the key "ts" of the json file.

    ###
    Sample input:
            [
                "conversation_id": "125",
                "List of text": [
                    [
                        "ts": "2020-05-03T08:14:12.575000",
                        "text": "Hi everyone, I'm new to discord and programming and looking for resources to learn python, Do you guys have any suggestions?"
                    ],
                    
                    [
                        "ts": "2020-05-03T08:14:23.490000",
                        "text": "Hi, we have a discord channel related to essential python resources. Maybe @Shira has some more advice for you."
                    ]
                ]
            ]

    Message:
    "{message}"
    
    
    Return a JSON object with a list of identified entities in the following format:
    {{"entities": ["Entity1", "Entity2", ...]}}
    """
    entities = """
    ***
    - APPLICATION
    - PROGRAMMING LANGUAGE
    - VERSION
    - ALGORITHM
    - OPERATION SYSTEM
    - DEVICE
    - ERROR NAME
    - USER NAME
    - DATA STRUCTURE
    - DATA TYPE
    - LIBRARY
    - LIBRARY CLASS
    - USER CLASS
    - LIBRARTY VARIABLE
    - USER VARIABLE
    - LIBRARTY FUNCTION
    - USER FUNCTION NAME
    - FILE TYPE
    - FILE NAME
    - UI ELEMENT
    - WEBSITE
    - ORGANIZATION
    - LICENSE
    - HTML/XML TAG NAME
    - VALUE
    - IN LINE CODE
    - OUTPUT BLOCK
    - KEYBOARD INPUT
    ***
    """
    response = query_mixtral(prompt)
    try:
        return json.loads(response).get("entities", [])
    except json.JSONDecodeError:
        return []

# Intent Classification
def classify_intent(message):
    """
    Calls Mixtral to classify the intent of the question in conversation message.
    """
    prompt = f"""Act as an NLP expert in analyzing conversations in Q/A computer programming chatrooms. Your task will be to determine the intent(s) behind a given question in a conversation based on a predefined list of intent categories. 
    The list of intent categories is: {intents}

    Based on the provided list and the given question in a conversation, return all applicable intents. 
    Other than the applicable intents, do not provide any justification! If there are no applicable intents,  return “NOT APPLICABLE”! 
    For example if you identify 'LEARNING' in the given question in a conversation, the output will be : [LEARNING]
    For example if you identify none of the predefined intents in the input question in a conversation, the output will be: [NOT APPLICABLE]

    Definition of intents:

    API USAGE:
    This category subsumes questions of the types How to implement something and Way of using something, as well as the category How-to, and Interaction of API Classes. The posts falling into this category contain questions asking for suggestions on how to implement some functionality or how to use an API. The questioner is asking for concrete instructions. 

    DISCREPANCY:
    This question category contains the categories Do not work, Discrepancy, What is the Problem...?, as well as Why (non-working code, errors, or unexpected behaviour). The posts in this category contain questions about problems and unexpected behaviour of code snippets, and the questioner has no clue how to solve them. 

    ERRORS:
    This question category is equivalent to the category Error and Exception Handling. Furthermore, it overlaps with the category Why (non-working code, errors, or unexpected behaviour) Similar to the previous category, posts in this category deal with the problems of exceptions and errors. Often, the questioner posts an exception and the stack trace and asks for help in fixing an error or understanding what the exception means. 

    REVIEW:
    This category merges the categories Decision Help and Review, the category Better Solution, and What (concepts, as well as asking for help to make a decision into the question category What), as well as How/Why something works (understanding, reading, explaining, and checking into the category How/Why something works). Questioners of these posts ask for better solutions or reviews of their code snippets. Often, they also ask for best practice approaches or ask for help making decisions, for instance, about which API to select. 

    CONCEPTUAL:
    This category is equivalent to the category conceptual and subsumes the categories Why...? and Is it possible...? Furthermore, it merges the categories What (concepts, as well as asking for help to make a decision into the question category What) and How/Why something works (understanding, reading, explaining, and checking into the category How/Why something works). The posts in this category consist of questions about the limitations of an API and API behaviour, as well as about understanding concepts, such as design patterns or architectural styles, and background information about some API functionality. 

    API CHANGE:
    This question category is equivalent to the categories Version and API Changes. These posts contain questions that arise due to changes in an API or due to compatibility issues between different versions of an API. 

    LEARNING:
    This category merges the categories of Learning a Language/Technology and Tutorials/Documentation. In these posts, the questioners ask for documentation or tutorials to learn a tool or language. In contrast to the first category, they do not aim at asking for a solution or instructions on how to do something. Instead, they aim at asking for support to learn on their own. 


    Message:
    "{message}"

    Return a JSON object with the intent classification:
    {{"intent": "CATEGORY"}}
    """
    intents = """
    - API USAGE
    - DISCREPANCY
    - ERRORS
    - REVIEW
    - CONCEPTUAL
    - API CHANGE
    - LEARNING
    """
    response = query_mixtral(prompt)
    try:
        return json.loads(response).get("intent", "UNKNOWN")
    except json.JSONDecodeError:
        return "UNKNOWN"

# Resolution Classification
def classify_resolution_status(conversation):
    """
    Calls Mixtral to determine whether the question received a valid answer.
    Uses the exact prompt from "1.3-mixtral_resolution_detector.py"
    """
    prompt = f"""Act as an NLP expert in analyzing conversations in Q/A computer programming chatrooms. Your task will be to determine whether the question in a given conversation has been properly addressed/solved.

    Instructions:
    If the question in the conversation is solved, the output should be: [SOLVED]
    If the question is not solved, the output should be: [NOT SOLVED]

    Criteria for determining the resolution status:
    [SOLVED] if the questioner acknowledges that their problem is resolved based on the answers received.
    [NOT SOLVED] if the questioner did not receive any answers related to their question.
    [NOT SOLVED] if the answers received do not address the question.
    [SOLVED] if the answers received are relevant and likely solve the problem, even if the questioner did not explicitly acknowledge it.

Conversation:
{conversation}

Return the result in JSON format:
{{"resolution_status": "SOLVED" or "UNSOLVED"}}
"""
    response = query_mixtral(prompt)
    try:
        return json.loads(response).get("resolution_status", "NOT SOLVED")
    except json.JSONDecodeError:
        return "NOT SOLVED"

# Apply Labeling
def apply_labeling(df):
    """
    Applies Mixtral-based labeling to extract entities, classify intent,
    and determine resolution status for each conversation.
    """
    labeled_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Messages"):
        conversation_text = row["conversation"]
        question_text = row["question"]

        # Mixtral Calls
        entities = extract_entities(question_text)
        intent = classify_intent(question_text)
        resolution_status = classify_resolution_status(conversation_text)

        # Append results
        labeled_data.append({
            "conversation_id": row["conversation_id"],
            "question": question_text,
            "conversation": conversation_text,
            "entities": entities,
            "intent": intent,
            "resolution_status": resolution_status
        })

    return labeled_data

# Save Labeled Data
def save_labeled_data(labeled_data, output_path):
    """
    Saves the labeled dataset to JSON format.
    """
    with open(output_path, "w") as f:
        for record in labeled_data:
            f.write(json.dumps(record) + "\n")
    print(f"[INFO] Labeled data saved to {output_path}")

# Main Execution
def main():
    print("[INFO] Loading preprocessed data...")
    df = load_data(INPUT_FILE)

    print("[INFO] Applying Mixtral-based labeling...")
    labeled_data = apply_labeling(df)

    print("[INFO] Saving labeled dataset...")
    save_labeled_data(labeled_data, OUTPUT_FILE)

    print("[SUCCESS] Labeling completed!")

if __name__ == "__main__":
    main()