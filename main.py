from src.graph import run_research_agent

from src.logger import save_research_result

question = "Should I use Kafka or RabbitMQ for a high-throughput event pipeline?"

answer = run_research_agent(question)

filepath = save_research_result(question, answer)

print(f"\n{'='*60}")
print("FINAL ANALYSIS:")
print('='*60)
print(answer)
print(f"\nSaved to: {filepath}")