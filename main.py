import json
from agent import WorkingAgent

def load_data(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def run_agent():
    agent = WorkingAgent()

def main():
    agent = WorkingAgent()

if __name__ == "__main__":
    main()