import os
import json
from datetime import datetime

def save_research_result(question: str, answer: str) -> str: 
    """
    Save a research result to a JSON file for later review and eval use.
    Returns the filepath where it was saved.
    """
    os.makedirs("results", exist_ok=True)

    #Create a clean filename from the question
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    clean_question = question[:50].replace(" ", "_").replace("?"," ")
    filename = f"results/{timestamp}_{clean_question}.json"

    result = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer
    }

    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

    return filename

if __name__ == "__main__":
    save_research_result("Q1","A1")