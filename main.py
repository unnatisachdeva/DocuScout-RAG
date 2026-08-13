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


embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
llm = init_chat_model("google_genai:gemini-2.5-flash-lite")
#retrieve data from chroma 
# vectorstore= Chroma(
#     persist_directory="chroma_db",
#     embedding_function=embedding_model
# )

# Create retriever
def get_retriever(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    return retriever


#prompt template 
prompt= ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use ONLY the provided context to answer the question. If you don't know the answer, say 'I could not find in the document' and do not make up an answer."),
    ("human", "Context: {Context}\n\nQuestion: {question}")
])

#ask question 

def ask_question(vectorstore, query):

    retriever = get_retriever(vectorstore)

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    # Combine chunks
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Create prompt
    final_prompt = prompt.invoke({
        "Context": context,
        "question": query
    })

    # Generate answer
    response = llm.invoke(final_prompt)

    return response.content

