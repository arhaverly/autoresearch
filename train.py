from openai import OpenAI
from dotenv import load_dotenv
import os


import experiment


def test_connection():
    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-5.4",
        input="Hello!"
    )

    print(response.output[0].content[0].text)


'''
What am I trying to do here?

I want an AI to automatically learn and improve an experiment.

What is that experiment?

Anything in experiment.py

The output should be put into output.csv


So the agent should run experiment.py, read the output.csv, then update experiment.py.
'''
def query_AI(program, config):
    with open(program, "r", encoding="utf-8") as f:
        instructions = f.read()

    with open(config, "r", encoding="utf-8") as f:
        config_json = f.read()

    instructions += config_json

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model="gpt-5.4",
        input=instructions
    )

    raw = response.output[0].content[0].text

    # Split into parts
    before, after = raw.split("START_JSON", 1)
    json_part, _ = after.split("END_JSON", 1)

    # Clean up
    explanation = before.strip()
    json_str = json_part.strip()

    with open(config, "w", encoding="utf-8") as f:
        f.write(json_str)
    
    with open("experiment_results.md", "a", encoding="utf-8") as f:
        f.write(f"Explanation: {explanation}")



def main():
    while True:
        query_AI('program.md', 'config.json')
        experiment.main()







if __name__ == '__main__':
    main()




























