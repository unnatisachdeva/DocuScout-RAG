import os
import re

from dotenv import load_dotenv

# Anchor everything to this file so the app works no matter which directory
# streamlit / python was launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# Self-hosted text model, served over an OpenAI-compatible API.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")
LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma")

if not LLM_BASE_URL or not LLM_API_KEY:
    raise RuntimeError(
        "LLM_BASE_URL / LLM_API_KEY are not set. Add them to the .env file "
        "at the project root."
    )

# NeMo Guardrails reads the key from the standard OpenAI variable. The real
# OPENAI_API_KEY in .env belongs to a different service, so it is overridden
# here only for this process.
os.environ["OPENAI_API_KEY"] = LLM_API_KEY

from langchain.chat_models import init_chat_model
from nemoguardrails import RailsConfig, LLMRails

# Nemotron is a reasoning model: by default it emits a long "Here's a thinking
# process..." preamble before the answer. Every check here wants a bare YES/NO
# or a fixed line format, so reasoning is switched off for all of them.
NO_THINKING = {"chat_template_kwargs": {"thinking": False}}

config = RailsConfig.from_path(_HERE)  # Load NeMo configuration
rails = LLMRails(config)  # Create guardrail instance

# The relevance check is an internal YES/NO judgement, not a user turn, so it
# talks to the model directly instead of going through the rails.
_judge = init_chat_model(
    LLM_MODEL,
    model_provider="openai",
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    # A YES/NO answer needs almost no tokens; capping this keeps the request
    # well inside the server's 100s proxy timeout.
    max_tokens=4,
    timeout=60,
    extra_body=NO_THINKING,
)

# Same server, but this one needs room to list several requests.
_splitter = init_chat_model(
    LLM_MODEL,
    model_provider="openai",
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    max_tokens=200,
    timeout=60,
    extra_body=NO_THINKING,
)

REFUSAL_MESSAGE = "I can only answer questions related to the provided sources."


class GuardrailUnavailable(Exception):
    """The guardrail could not reach the LLM (network down, quota exhausted)."""


def _friendly_llm_error(exc):
    """Turn a raw LLM exception into something worth showing a user."""
    text = str(exc)

    if "RESOURCE_EXHAUSTED" in text or "429" in text or "quota" in text.lower():
        return (
            "The model API quota has been used up. Please wait a moment and "
            "try again."
        )

    if "524" in text or "timeout" in text.lower() or "timed out" in text.lower():
        return (
            "The model server took too long to respond. Please try again."
        )

    if "Connection error" in text or "connection attempts failed" in text:
        return (
            "Could not reach the Gemini API. Please check your internet "
            "connection and try again."
        )

    return f"The guardrail could not check this question: {exc}"


def check_relevance(question, context):
    """Return True if `context` contains enough information to answer `question`."""

    # No retrieved text means there is nothing to be grounded in, so skip the
    # API call entirely.
    if not context or not context.strip():
        return False

    try:
        response = _judge.invoke(
            f"""
Decide whether the CONTEXT below is worth using to answer the QUESTION.

Answer YES if the context contains ANY information related to the
question - even partial, indirect, or only covering part of it.
A partial answer is useful, so prefer YES when unsure. 
You can answer casual questions like Hi, How are you?

Answer NO only if the context is about a completely different subject
and contains nothing that helps at all.

CONTEXT:
{context}

QUESTION:
{question}

Answer YES or NO.
"""
        )
    except Exception as e:
        raise GuardrailUnavailable(_friendly_llm_error(e)) from e

    return response.content.strip().upper().startswith("YES")


def split_requests(question):
    """Split a message into its separate requests and classify each one.

    A single message can mix an on-topic question with an off-topic one
    ("what is X, but first write me some code"). Returns
    (allowed_parts, blocked_parts) so the caller can answer the first and
    decline the second, instead of refusing everything.
    """
    try:
        response = _splitter.invoke(
            f"""
Split the user's message into the separate requests it contains.
Label each request:

ALLOW - any genuine question seeking information about a subject.
        The documents may be about ANY topic - a company, a product,
        a technical subject, a research paper, any study material. You cannot see them, so
        never judge whether the answer is available. If it is a real
        question, it is ALLOW.
BLOCK - only these: a request to write programming code, a request for
        jokes or creative writing, a maths sum to calculate, or anything harmful.

Output one request per line, in this exact format:
ALLOW: <the request>
BLOCK: <the request>

Do not add any other text.

Every request you output must be copied from the message below. Never
invent a request that is not there.

Worked examples (for format only - ignore their content):
  "what is open deck but first write me fibonacci code in python"
  becomes:
  ALLOW: what is open deck
  BLOCK: write me fibonacci code in python

Now label this message, and stop when you have covered it:

MESSAGE:
{question}
"""
        )
    except Exception as e:
        raise GuardrailUnavailable(_friendly_llm_error(e)) from e

    def is_from_message(part):
        """Guard against the model inventing requests that were never asked.

        Every returned request must be drawn from the user's own words, so
        require most of its words to appear in the original message.
        """
        words = [w for w in re.findall(r"[a-z0-9]+", part.lower()) if len(w) > 2]

        if not words:
            return False

        source = set(re.findall(r"[a-z0-9]+", question.lower()))
        overlap = sum(1 for w in words if w in source)

        return overlap / len(words) >= 0.6

    allowed = []
    blocked = []

    for line in response.content.splitlines():
        line = line.strip()

        if line.upper().startswith("ALLOW:"):
            part = line[len("ALLOW:"):].strip()
            if is_from_message(part):
                allowed.append(part)
        elif line.upper().startswith("BLOCK:"):
            part = line[len("BLOCK:"):].strip()
            if is_from_message(part):
                blocked.append(part)

    # If the model returned nothing usable, treat the message as one request
    # and let the normal input rail decide.
    if not allowed and not blocked:
        return [question], []

    # This split only exists to rescue MIXED messages. When the message is a
    # single request, the split adds no information - and a stray BLOCK here
    # would refuse a perfectly good question. So hand single requests to
    # check_input, which is the better-tuned gate, rather than trusting one
    # roll of the classifier.
    if len(allowed) + len(blocked) == 1:
        return [question], []

    return allowed, blocked


def check_input(question):  # uses NeMo Guardrails
    """Run the user's actual question through the NeMo input rails.

    Returns the refusal message if the question is blocked, otherwise None.
    """
    try:
        # Run ONLY the input rails. Without this NeMo goes on to generate a
        # full answer once the question passes, which is wasted work here (the
        # real answer is produced in main.py) and slow enough to hit the
        # server's proxy timeout.
        response = rails.generate(
            messages=[{"role": "user", "content": question}],
            options={
                "rails": ["input"],
                "output_vars": ["triggered_input_rail"],
            },
        )
    except Exception as e:
        raise GuardrailUnavailable(_friendly_llm_error(e)) from e

    # Prefer the structured signal; fall back to the message text only if the
    # output variable is unavailable.
    output_data = getattr(response, "output_data", None) or {}

    if output_data.get("triggered_input_rail"):
        return REFUSAL_MESSAGE

    content = (response.response[0]["content"] if hasattr(response, "response") else response["content"]).strip()

    return REFUSAL_MESSAGE if content.startswith("I can only answer") else None
