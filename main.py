'''main.py is only responsible for:

taking a vectorstore
retrieving documents
creating the prompt
calling the LLM'''

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings #embeddings
from langchain_chroma import Chroma #retrieve db
from langchain_core.prompts import ChatPromptTemplate #prompt template
from langchain.chat_models import init_chat_model
from guardrails.rag_guardrail import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    NO_THINKING,
    REFUSAL_MESSAGE,
    GuardrailUnavailable,
    _friendly_llm_error,
    check_input,
    check_relevance,
    split_requests
)
from query_rewriter import rewrite_query


# Embeddings stay on Gemini; only the text model changed.
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
llm = init_chat_model(
    LLM_MODEL,
    model_provider="openai",
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    # Without this the answer arrives wrapped in the model's chain of thought.
    extra_body=NO_THINKING,
)
#retrieve data from chroma 
# vectorstore= Chroma(
#     persist_directory="chroma_db",
#     embedding_function=embedding_model
# )


# def get_retriever(vectorstore):

#     # k=3 was too narrow for a whole crawled website: a page-specific question
#     # ("regulations in Alaska") could easily miss its one relevant chunk.
#     retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k": 8,
#             "fetch_k": 40,
#             "lambda_mult": 0.5
#         }
#     )

#     return retriever


#prompt template 
prompt= ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use ONLY the provided context to answer the question. If you don't know the answer, say 'I could not find in the document' and do not make up an answer."),
    ("human", "Context: {Context}\n\nQuestion: {question}")
])

#ask question 

def ask_question(retriever,llm, query, chat_history=None):
    # --------------------------------
    # 1. Input guardrail
    # --------------------------------

    # A message can mix an on-topic question with an off-topic one, so split
    # it up and keep only the parts worth answering.
    try:
        allowed_parts, blocked_parts = split_requests(query)
    except GuardrailUnavailable as e:
        return str(e)

    if not allowed_parts:
        return REFUSAL_MESSAGE

    answerable = " ".join(allowed_parts)

    # Note explaining anything that was dropped, appended to the answer.
    declined_note = ""

    if blocked_parts:
        declined_note = (
            "\n\n_Note: I could not help with "
            + "; ".join(f'"{part}"' for part in blocked_parts)
            + " — I can only answer questions about the provided sources._"
        )

    try:
        refusal = check_input(answerable)
    except GuardrailUnavailable as e:
        return str(e)

    if refusal:
        return refusal

    # --------------------------------
    # 2. Retrieve relevant chunks
    # --------------------------------
    rewritten_query= rewrite_query(llm, answerable, chat_history)

    docs = retriever.invoke(rewritten_query)

    # Combine relevant chunks
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # --------------------------------
    # 4. Check whether context is
    #    actually relevant
    # --------------------------------

    try:
        is_relevant = check_relevance(
            answerable,
            context
        )
    except GuardrailUnavailable as e:
        return str(e)

    if not is_relevant:

        return (
            "I could not find the answer in the provided sources."
            + declined_note
        )

    # --------------------------------
    # 5. Generate final answer
    # --------------------------------

    # Create prompt
    final_prompt = prompt.invoke({
        "Context": context,
        "question": answerable
    })

    # Generate answer
    try:
        response = llm.invoke(final_prompt)
    except Exception as e:
        return _friendly_llm_error(e)

    return response.content + declined_note

