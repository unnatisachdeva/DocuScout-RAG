#test query rewriter 
print("Starting test...")

from main import llm

print("LLM imported...")

from query_rewriter import rewrite_query

print("Rewriter imported...")

# Test 1: vague follow-up should get resolved using history
history = [
    "user: What is a CNN?",
    "assistant: A CNN is a deep learning model used for image classification."
]
print("Calling rewrite_query...")
result = rewrite_query(llm, "how does it compare to RNNs?", history)
print("Rewritten:", result)
# Expect something like: "How does CNN compare to RNN?"
# NOT the original vague "how does it compare to RNNs?"

# Test 2: no history — should still return something sensible

print("Running Test 2...")
result2 = rewrite_query(llm, "wat is imag classifcation", None)
print("Rewritten:", result2)
# Expect typo fixed: "What is image classification?"

