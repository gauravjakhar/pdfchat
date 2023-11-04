import shutil
import tempfile
import os
from ingestorandgpt import FileIngestorGPT
import openai
import streamlit as st
from openai.error import AuthenticationError


# Function to load the OpenAI API key
def load_openai_api_key():
    # Prompt for the API key if not already set in session state
    if 'openai_api_key' not in st.session_state or not st.session_state['openai_api_key']:
        api_key = st.text_input("Enter your OpenAI API key", type="password", key="temp_api_key")
        if api_key:
            try:
                # Test the API key by making a dummy request
                openai.Completion.create(engine="davinci", prompt="test", max_tokens=5, api_key=api_key)
                st.session_state['openai_api_key'] = api_key
            except AuthenticationError:
                # If authentication fails, prompt again
                st.error("Incorrect API key provided. Please enter the correct OpenAI API key.")
                st.session_state['openai_api_key'] = None  # Reset the key in session state
                return None
    return st.session_state.get('openai_api_key')


# Prompt for the API key at the start of the app
openai_api_key = load_openai_api_key()
if not openai_api_key:
    st.error("Please enter your OpenAI API key.")
    st.stop()

# Set the title for the Streamlit app
st.title("Chat with PDF - 🤖 🔗")


# Function to save uploaded file to a temporary directory
def save_uploaded_file(uploaded_file):
    # Create a new temporary directory for each file/session
    if 'file_paths' not in st.session_state:
        st.session_state['file_paths'] = {}

    # Generate a temporary file name
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        # Write the uploaded file's contents to the temporary file
        shutil.copyfileobj(uploaded_file, tmp_file)
        st.session_state['file_paths'][uploaded_file.name] = tmp_file.name

    return st.session_state['file_paths'][uploaded_file.name]


# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    file_path = save_uploaded_file(uploaded_file)

    # Process the file using the temporary file path
    ingestor = FileIngestorGPT(file_path)
    ingestor.handlefileandingestGPT()


# At the end of the session or when you're done with the file, clean up the temporary files
def clean_up_files():
    if 'file_paths' in st.session_state:
        for file_path in st.session_state['file_paths'].values():
            if os.path.exists(file_path):
                os.remove(file_path)
