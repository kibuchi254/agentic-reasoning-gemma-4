from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="gemma4:latest",
    messages=[
        {
            "role": "system",
            "content": "You are an agentic SaaS AI assistant."
        },
        {
            "role": "user",
            "content": "Create an invoicing workflow for SMEs."
        }
    ]
)

print(response.choices[0].message.content)