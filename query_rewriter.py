from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

from guardrails.rag_guardrail import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    NO_THINKING,
)

REWRITE_PROMPT= ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite questions into standalone search queries for a document "
     "retrieval system.\n"
     "\n"
     "Rules:\n"
     "- Replace pronouns and vague references using the chat history.\n"
     "- Fix spelling mistakes.\n"
     "- Keep it under 15 words.\n"
     "- NEVER answer the question. You do not know the answer, and any "
     "answer you give will be wrong.\n"
     "- Output the rewritten question ONLY, with no explanation.\n"
     "\n"
     "Examples:\n"
     "History: user: What is a CNN? / assistant: A CNN is a deep learning "
     "model.\n"
     "Question: how does it compare to RNNs?\n"
     "Output: How does a CNN compare to an RNN?\n"
     "\n"
     "History: (none)\n"
     "Question: wat is imag classifcation\n"
     "Output: What is image classification?\n"
     "\n"
     "History: user: Do you ship to Alaska? / assistant: Yes, we do.\n"
     "Question: what are the rules there\n"
     "Output: What are the shipping rules for Alaska?"
     ),
     ("human", "History:\n{history}\n\nQuestion: {question}\n\nOutput:")
])

# A rewritten query is one short line, so cap it. Reusing the uncapped answering
# model let this call run for minutes; the short timeout also means a dead
# server fails in seconds instead of stalling every chat message.
_rewriter = init_chat_model(
    LLM_MODEL,
    model_provider="openai",
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    max_tokens=60,
    timeout=20,
    # No retries: when the server is down Cloudflare answers with
    # "retry_after: 120", and the client obeys it - turning a fast failure into
    # a two-minute stall on every chat message. The rewrite is optional, so
    # fail immediately and fall back to the original query.
    max_retries=0,
    extra_body=NO_THINKING,
)


def rewrite_query(llm, query, chat_history=None):
    """Rewrite `query` into a standalone search query.

    Falls back to the original query if the model is unavailable, so retrieval
    still works - but the failure is reported instead of silently swallowed.
    """
    history_text = "\n".join(chat_history[-4:]) if chat_history else "(none)"

    prompt = REWRITE_PROMPT.invoke({
        "history": history_text,
        "question": query
    })

    try:
        response = _rewriter.invoke(prompt)

    except Exception as e:
        # Loud on purpose: without this, a dead server looks exactly like a
        # query that simply needed no rewriting.
        print(f"[query_rewriter] rewrite failed, using original query: {e}")
        return query

    rewritten = response.content.strip()

    if not rewritten:
        print("[query_rewriter] empty rewrite, using original query")
        return query

    # Small models sometimes wrap the result in quotes or prefix a label
    # such as "Rewritten query: ...".
    rewritten = rewritten.strip('"').strip()

    head, sep, rest = rewritten.partition(":")

    if sep and len(head.split()) <= 3 and rest.strip():
        rewritten = rest.strip()

    return rewritten or query
