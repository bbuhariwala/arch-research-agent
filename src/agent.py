import anthropic
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a senior software architect who specializes in 
analyzing technical architecture decisions. Given a question, you identify 
key tradeoffs, consider operational concerns, and give structured analysis. 
You are concise, direct, and always explain your reasoning."""

def ask_claude(question: str, conversation_history: list = None) -> str: 
    """
    Send a question to claude and get a response.
    conversation_history is an optional list of previous messages.
    """
    client = anthropic.Anthropic()
    messages = conversation_history or []
    messages.append({
        "role": "user",
        "content" : question
    })

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    return response.content[0].text

if __name__ == "__main__":
    question = "Should I use Kafka or RabbitMQ for a high-throughput event pipeline?"
    answer = ask_claude(question)
    print("Answer:\n", answer)
