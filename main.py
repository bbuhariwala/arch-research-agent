from src.graph import run_research_agent
from src.logger import save_research_result

questions = [
    "Should I use Kafka or RabbitMQ for a high-throughput event pipeline?",
    "Should I use PostgreSQL or MongoDB for a user profile service?",
    "Should I use REST or GraphQL for a mobile app API?"
]

questions = [
    "Should I use Kafka or RabbitMQ for a high-throughput event pipeline?",
]

for question in questions:
    answer = run_research_agent(question)
    filepath = save_research_result(question, answer)
    print(f"\n{'='*60}")
    print("FINAL ANALYSIS:")
    print('='*60)
    print(answer)
    print(f"\nSaved to: {filepath}")
    print("\n" + "="*60 + "\n")