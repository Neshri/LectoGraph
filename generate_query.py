import requests
import yaml
import os

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
OLLAMA_URL = config.get("ollama_url", "http://127.0.0.1:11434")
LLM_MODEL = config.get("summary_model", "gemma4:31b")

def ollama_complete(
    model=LLM_MODEL, prompt=None, system_prompt=None, history_messages=None, **kwargs
) -> str:
    """
    Sends a completion request to a local Ollama instance via its REST API.
    """
    if history_messages is None:
        history_messages = []
        
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        **kwargs
    }
    
    response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload)
    response.raise_for_status()
    result = response.json()
    
    return result["message"]["content"]

if __name__ == "__main__":
    description = "Detta dataset innehåller en kunskapsgraf baserad på djupgående sammanställningar och analyser av över 400 inspelade videor med IT-lektioner och förklaringar. Materialet täcker både vad som visas på skärmen (tidslinjer, fönster, terminalkommandon, kod) och vad som sägs i ljudtranskriptionerna."
    prompt = f"""
    Givet följande beskrivning av ett dataset:

    {description}

    Vänligen identifiera 5 potentiella användare som skulle interagera med detta dataset. För varje användare, lista 5 uppgifter de skulle utföra med datasetet. Sedan, för varje kombination av (användare, uppgift), generera 5 frågor som kräver en övergripande förståelse av hela datasetet.

    Returnera resultatet i följande struktur:
    - Användare 1: [beskrivning av användare]
        - Uppgift 1: [beskrivning av uppgift]
            - Fråga 1:
            - Fråga 2:
            - Fråga 3:
            - Fråga 4:
            - Fråga 5:
        - Uppgift 2: [beskrivning av uppgift]
            ...
        - Uppgift 5: [beskrivning av uppgift]
    - Användare 2: [beskrivning av användare]
        ...
    - Användare 5: [beskrivning av användare]
        ...
    """

    print("Genererar frågor via Ollama...")
    try:
        result = ollama_complete(model=LLM_MODEL, prompt=prompt)

        file_path = "./queries.txt"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(result)

        print(f"Frågor sparade i {file_path}")
    except Exception as e:
        print(f"Ett fel uppstod: {e}")
