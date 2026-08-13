'''bridge between the ingestion.py and main.py files. It is responsible for the UI and user interaction.
'''
import os
import tempfile

import streamlit as st

from ingestion import create_vectorstore
from main import ask_question


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 PDF RAG Chatbot")

st.write(
    "Upload a PDF and ask questions about its contents."
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Upload your PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    process_button = st.button(
        "Process PDF",
        use_container_width=True
    )


# --------------------------------------------------
# Process PDF
# --------------------------------------------------

if process_button:

    if uploaded_file is None:

        st.warning("Please upload a PDF first.")

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
                    f.write(uploaded_file.getbuffer())


                # Chroma database location
                chroma_dir = os.path.join(
                    temp_dir,
                    "chroma_db"
                )


                # Create vector store
                vectorstore, page_count, chunk_count = (
                    create_vectorstore(
                        pdf_path,
                        chroma_dir
                    )
                )


                # Store vector database
                st.session_state.vectorstore = vectorstore

                # Store PDF name
                st.session_state.pdf_name = uploaded_file.name

                # Clear old chat
                st.session_state.messages = []


                st.success("PDF processed successfully!")

                st.info(
                    f"📄 Pages: {page_count} | "
                    f"🧩 Chunks: {chunk_count}"
                )


            except Exception as e:

                st.error(
                    f"Error processing PDF: {e}"
                )


# --------------------------------------------------
# Current PDF
# --------------------------------------------------

if st.session_state.pdf_name:

    st.success(
        f"Currently chatting with: "
        f"**{st.session_state.pdf_name}**"
    )


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------

query = st.chat_input(
    "Ask something about your PDF..."
)


if query:

    # Make sure PDF exists
    if st.session_state.vectorstore is None:

        st.warning(
            "Please upload and process a PDF first."
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