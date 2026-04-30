TOOLS =[
    {
        "name": "search_web",
        "description": """Search the web for information about architecture decisions, 
        technical tradeoffs, and engineering discussions. Use this when you need 
        current information from Hacker News, GitHub, or technical blogs to support 
        your analysis. Search for specific technical comparisons, real-world experiences, 
        and community opinions.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific — include technology names and the specific aspect you want to research e.g. 'Kafka vs RabbitMQ throughput benchmarks' or 'RabbitMQ operational complexity production'"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return. Default 5, max 10.",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
]