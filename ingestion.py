# '''ingestion.py creates the vectorstore
# It says:

# "Give me Documents(be it the doc come from pdf or website), and I'll handle splitting → embeddings → vector DB. '''
# from dotenv import load_dotenv
# load_dotenv()

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma


# def create_vectorstore(docs, persist_directory):

#     # Load PDF
#     # loader = PyPDFLoader(pdf_path)
#     # docs = loader.load()

#     # Split into chunks
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )

#     chunks = splitter.split_documents(docs)

#     # Embedding model
#     embedding_model = GoogleGenerativeAIEmbeddings(
#         model="gemini-embedding-2"
#     )

#     # Create vector database
#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embedding_model,
#         persist_directory=persist_directory
#     )

#     return vectorstore, len(docs), len(chunks)



'''ingestion.py creates the vectorstore

Input:
    Documents (PDF or Website)

Pipeline:
    Documents
        ↓
    Chunking
        ↓
    Batch insertion
        ↓
    Embeddings
        ↓
    Chroma
'''

import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


def create_vectorstore(
    docs,
    persist_directory
):

    # --------------------------------
    # Split documents
    # --------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        docs
    )


    print(
        f"Documents: {len(docs)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )


    # --------------------------------
    # Embedding model
    # --------------------------------

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )


    # --------------------------------
    # Create empty Chroma DB
    # --------------------------------

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )


    # --------------------------------
    # Add chunks in batches
    # --------------------------------

    batch_size = 20

    total_batches = (
        len(chunks) + batch_size - 1
    ) // batch_size


    for start in range(
        0,
        len(chunks),
        batch_size
    ):

        batch_number = (
            start // batch_size
        ) + 1


        batch = chunks[
            start:start + batch_size
        ]


        print(
            f"Adding batch "
            f"{batch_number}/{total_batches}"
        )


        retries = 3


        while retries > 0:

            try:

                vectorstore.add_documents(
                    batch
                )

                break


            except Exception as e:

                if (
                    "429" in str(e)
                    or
                    "RESOURCE_EXHAUSTED"
                    in str(e)
                ):

                    print(
                        "Embedding quota reached."
                    )

                    print(
                        "Waiting 45 seconds..."
                    )


                    time.sleep(45)

                    retries -= 1

                else:

                    raise e


        if retries == 0:

            raise Exception(
                "Embedding failed after retries"
            )


    print(
        "Vectorstore creation completed."
    )


    return (
        vectorstore,
        len(docs),
        len(chunks)
    )