import os
import openai
import streamlit as st
from openai.error import AuthenticationError
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from streamlit_chat import message
import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'


# Function to load the OpenAI API key
def load_openai_api_key():
    # Check if the API key is set in environment variables
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    # If not found, prompt the user to enter their API key in the app
    if api_key:
        # Set the API key in the environment variable for this session
        os.environ["OPENAI_API_KEY"] = api_key
        try:
            openai.api_key = api_key
            openai.Completion.create(engine="davinci", prompt="test", max_tokens=5)
            # If the key is valid, set it in the environment variable for this session
            os.environ["OPENAI_API_KEY"] = api_key
        except AuthenticationError:
            # If authentication fails, delete the incorrect key from session state and prompt again
            st.error("Incorrect API key provided. Please enter the correct OpenAI API key.")
            st.session_state.pop('api_key', None)  # This will clear the input box for re-entry
            return None

    return api_key


# Prompt for the API key at the start of the app
openai_api_key = load_openai_api_key()
if not openai_api_key:
    st.error("Please enter your OpenAI API key.")
    st.stop()


class FileIngestorGPT:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def handlefileandingestGPT(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(file_path=tmp_file_path)
        data = loader.load()

        # Create embeddings using Sentence Transformers
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # Create a FAISS vector store and save embeddings
        db = FAISS.from_documents(data, embeddings)
        db.save_local(DB_FAISS_PATH)

        # Create a conversational chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            retriever=db.as_retriever())

        # Function for conversational chat
        def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]

        # Initialize chat history
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        # Initialize messages
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me(LLAMA2) about " + self.uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]

        # Create containers for chat history and user input
        response_container = st.container()
        container = st.container()

        # User input form
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to PDF data 🧮", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        # Display chat history
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
