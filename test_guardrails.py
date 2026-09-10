import os

from dotenv import load_dotenv

load_dotenv()

# nemoguardrails 0.23 talks to Gemini through its OpenAI-compatible endpoint,
# so the Google key has to be visible as OPENAI_API_KEY.
os.environ["OPENAI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

from nemoguardrails import RailsConfig
from nemoguardrails import LLMRails


# Load guardrail configuration
config = RailsConfig.from_path(
    "guardrails"
)

# Create NeMo Guardrails
rails = LLMRails(config)


# Test question
response = rails.generate(
    messages=[
        {
            "role": "user",
            "content": "Write a C++ program to reverse a string."
        }
    ]
)

print("ANSWER:")
print(response["content"])