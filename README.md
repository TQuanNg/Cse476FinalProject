# CSE 476 Final Project – Intelligent Question Answering Agent

## Author
**Tan Nguyen**  
Course: CSE 476  

## Folder Structure
├── Generate_answer_template.py # Main entry file

├── Agent.py # Core agent logic

├── inference_techniques.py # Reasoning strategies

├── utils.py # LLM wrapper and helpers


## File Descriptions

### Generate_answer_template.py
 **This is the file to reproduce result**
**Main entry script for generating answers.**


Note: The loop strictly adheres to the input/output JSON question format.
For example, only when the output file have this JSON output format, then the agent loop will run: 
```python
def is_placeholder(answer_text: str) -> bool:
    return answer_text.startswith("Placeholder answer")
```

Important variables:
```python
INPUT_FILE_PATH
OUTPUT_FILE_PATH
START_INDEX
END_INDEX
```
START_INDEX, END_INDEX, are for a customized range

### Agent.py
Contains the Agent class.
Responsibilities:

- Classifies input questions

- Selects reasoning strategy

Calls appropriate inference method

Returns finalized answer

### Inference_techniques.py
Contains all reasoning strategies including:

- Chain-of-Thought (CoT)

- Continuation prompting

- Self-refinement loop

- Analogical reasoning

- ReAct prompting

- Self-consistency voting
