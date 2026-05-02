from src.agent import run_research_agent

answer = run_research_agent(
    "Should I use Kafka or RabbitMQ for a high-throughput event pipeline?"
)

print(answer)