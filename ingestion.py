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

import os
import re
import time

from dotenv import load_dotenv

# This module currently only works because main.py happens to be imported
# first; load the keys here too so it stands on its own.
load_dotenv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Must match the model used for queries in main.py. Indexing with one model and
# querying with another silently wrecks retrieval: the vectors have the same
# dimension, so nothing errors, but the distances are meaningless.
EMBEDDING_MODEL = "gemini-embedding-001"


def _retry_delay_from_error(message, attempt):
    """Seconds to wait before retrying, honouring the server's own hint."""
    match = re.search(r"retryDelay[\"']?\s*:\s*[\"']?(\d+(?:\.\d+)?)", message)

    if match:
        return float(match.group(1)) + 2

    # Otherwise back off exponentially: 15s, 30s, 60s, 120s...
    return min(15 * (2 ** attempt), 120)


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
        model=EMBEDDING_MODEL
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


        max_attempts = 6

        last_error = None


        for attempt in range(max_attempts):

            try:

                vectorstore.add_documents(
                    batch
                )

                last_error = None

                break


            except Exception as e:

                message = str(e)

                last_error = e


                is_rate_limited = (
                    "429" in message
                    or
                    "RESOURCE_EXHAUSTED" in message
                    or
                    "quota" in message.lower()
                )


                if not is_rate_limited:

                    raise


                # The daily quota cannot be waited out, so stop immediately
                # instead of sleeping through five pointless retries.
                if "PerDay" in message:

                    raise RuntimeError(
                        "The daily embedding quota for this Google project is "
                        "exhausted. It resets at midnight Pacific time, or you "
                        "can enable billing to remove the limit. "
                        f"Indexed {start} of {len(chunks)} chunks before stopping."
                    ) from e


                wait = _retry_delay_from_error(message, attempt)

                print(
                    f"Embedding rate limit hit. Waiting {wait:.0f}s "
                    f"(attempt {attempt + 1}/{max_attempts})..."
                )

                time.sleep(wait)


        if last_error is not None:

            raise RuntimeError(
                f"Embedding failed after {max_attempts} attempts on batch "
                f"{batch_number}/{total_batches}. Last error: {last_error}"
            ) from last_error


        # Stay under the per-minute request limit on large sites.
        time.sleep(1)

    print(
        "Vectorstore creation completed."
    )


    return (
        vectorstore,
        len(docs),
        len(chunks)
    )