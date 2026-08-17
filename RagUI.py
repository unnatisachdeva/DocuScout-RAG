'''Bridge between ingestion.py and main.py.
Responsible for UI and user interaction.
'''

import os
import tempfile

import streamlit as st

from ingestion import create_vectorstore
from main import ask_question
from langchain_community.document_loaders import PyPDFLoader
from web_loader.crawler import crawl_website


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 RAG Chatbot")

st.write(
    "Upload a PDF or enter a website URL and ask questions about its contents."
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "source_name" not in st.session_state:
    st.session_state.source_name = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📚 Choose Source")

    source_type = st.radio(
        "What do you want to chat with?",
        ["PDF", "Website"]
    )


    # ----------------------------------------------
    # PDF option
    # ----------------------------------------------

    if source_type == "PDF":

        uploaded_file = st.file_uploader(
            "Choose a PDF",
            type=["pdf"]
        )

        process_button = st.button(
            "Process PDF",
            use_container_width=True
        )


    # ----------------------------------------------
    # Website option
    # ----------------------------------------------

    else:

        website_url = st.text_input(
            "Enter website URL",
            placeholder="https://example.com"
        )

        process_button = st.button(
            "Process Website",
            use_container_width=True
        )


# --------------------------------------------------
# Process selected source
# --------------------------------------------------

if process_button:

    # ==================================================
    # PDF
    # ==================================================

    if source_type == "PDF":

        if uploaded_file is None:

            st.warning(
                "Please upload a PDF first."
            )

        else:

            with st.spinner(
                "Processing PDF... Please wait."
            ):

                try:

                    # Temporary directory
                    temp_dir = tempfile.mkdtemp()

                    pdf_path = os.path.join(
                        temp_dir,
                        uploaded_file.name
                    )

                    # Save uploaded PDF
                    with open(pdf_path, "wb") as f:

                        f.write(
                            uploaded_file.getbuffer()
                        )


                    # Chroma database location
                    chroma_dir = os.path.join(
                        temp_dir,
                        "chroma_db"
                    )


                    # Load PDF
                    loader = PyPDFLoader(
                        pdf_path
                    )

                    pdf_docs = loader.load()


                    # Create vector store
                    vectorstore, page_count, chunk_count = (
                        create_vectorstore(
                            pdf_docs,
                            chroma_dir
                        )
                    )


                    # Store vector database
                    st.session_state.vectorstore = vectorstore

                    # Store source name
                    st.session_state.source_name = (
                        uploaded_file.name
                    )

                    # Clear old chat
                    st.session_state.messages = []


                    st.success(
                        "PDF processed successfully!"
                    )

                    st.info(
                        f"📄 Pages: {page_count} | "
                        f"🧩 Chunks: {chunk_count}"
                    )


                except Exception as e:

                    st.error(
                        f"Error processing PDF: {e}"
                    )


    # ==================================================
    # WEBSITE
    # ==================================================

    elif source_type == "Website":

        if not website_url:

            st.warning(
                "Please enter a website URL first."
            )

        else:

            with st.spinner(
                "Crawling website and creating embeddings..."
            ):

                try:

                    # Crawl website
                    website_docs = crawl_website(
                        website_url,
                        max_pages=20
                    )


                    # Temporary Chroma directory
                    temp_dir = tempfile.mkdtemp()

                    chroma_dir = os.path.join(
                        temp_dir,
                        "chroma_db"
                    )


                    # Create vector store
                    vectorstore, doc_count, chunk_count = (
                        create_vectorstore(
                            website_docs,
                            chroma_dir
                        )
                    )


                    # Store vector database
                    st.session_state.vectorstore = vectorstore

                    # Store website name
                    st.session_state.source_name = (
                        website_url
                    )

                    # Clear old chat
                    st.session_state.messages = []


                    st.success(
                        "Website processed successfully!"
                    )

                    st.info(
                        f"🌐 Pages: {doc_count} | "
                        f"🧩 Chunks: {chunk_count}"
                    )


                except Exception as e:

                    st.error(
                        f"Error processing website: {e}"
                    )


# --------------------------------------------------
# Current source
# --------------------------------------------------

if st.session_state.source_name:

    st.success(
        f"Currently chatting with: "
        f"**{st.session_state.source_name}**"
    )


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# Chat input
# --------------------------------------------------

query = st.chat_input(
    "Ask something about your source..."
)


if query:

    # Make sure source exists
    if st.session_state.vectorstore is None:

        st.warning(
            "Please process a PDF or website first."
        )

        st.stop()


    # Display user message
    with st.chat_message("user"):

        st.markdown(query)


    st.session_state.messages.append({
        "role": "user",
        "content": query
    })


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = ask_question(
                st.session_state.vectorstore,
                query
            )

            st.markdown(answer)


    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })