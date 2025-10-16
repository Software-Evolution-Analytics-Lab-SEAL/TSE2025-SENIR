# Full Prompts

## Entity Extraction Prompt
```python
prompt = f"""As an expert in Named Entity Recognition (NER) specialized in analyzing chatroom discussions about computer programming concepts, your task is to identify tokens from the given block of messages of a question in a conversation that correspond to a provided list of entities delimited by *** {entities}. For each identified token, report the output in JSON as:
{"entities": ["Entity1", "Entity2", ...]}"""
```

### Entities List
```
***
- APPLICATION
- PROGRAMMING LANGUAGE
- VERSION
- ALGORITHM
- OPERATING SYSTEM
- DEVICE
- ERROR NAME
- USER NAME
- DATA STRUCTURE
- DATA TYPE
- LIBRARY
- LIBRARY CLASS
- USER CLASS
- LIBRARY VARIABLE
- USER VARIABLE
- LIBRARY FUNCTION
- USER FUNCTION NAME
- FILE TYPE
- FILE NAME
- UI ELEMENT
- WEBSITE
- ORGANIZATION
- LICENSE
- HTML/XML TAG NAME
- VALUE
- INLINE CODE
- OUTPUT BLOCK
- KEYBOARD INPUT
***
```

## Intent Classification Prompt
```python
prompt = f"""Act as an NLP expert in analyzing Q/A chatrooms. Given a question in a conversation and a predefined list of intent categories: {intents}, return all applicable intents in JSON:
{"intents": ["CATEGORY1", "CATEGORY2", ...]}"""
```

### Intents List
```
- API USAGE
- DISCREPANCY
- ERRORS
- REVIEW
- CONCEPTUAL
- API CHANGE
- LEARNING
```

## Resolution Classification Prompt
```python
prompt = f"""Act as an NLP expert in analyzing Q/A chatrooms. Determine whether the question has been solved. Return JSON:
{"resolution_status": "SOLVED"} or {"resolution_status": "UNSOLVED"}

Criteria:
- SOLVED: questioner acknowledges resolution or answers clearly solve the problem.
- UNSOLVED: no relevant answers or questioner did not acknowledge resolution.
"""
```
