from src.agent import ask_claude

answer = ask_claude(
    "Should I use Kafka or RabbitMQ for a high-throughput event pipeline?"
)

print(answer)