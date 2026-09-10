from dotenv import load_dotenv

load_dotenv()

from guardrails.rag_guardrail import check_relevance


context = """
Growth Natives is a digital marketing company.
It provides services such as SEO, PPC, content marketing,
social media marketing and web development.
"""


question = "What services does Growth Natives provide?"

result = check_relevance(
    question,
    context
)

print("Question:", question)
print("Guardrail result:", result)


question = "Write a C++ program to reverse a string."

result = check_relevance(
    question,
    context
)

print("\nQuestion:", question)
print("Guardrail result:", result)