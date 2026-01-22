import os

import google.generativeai as genai

from pdf_extractor import text_extractor

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

import streamlit as st

# Lets configure the models

# 1. LLM Model
gemini_key = os.getenv('Google_api_key1')
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-2.5-flash')

# 2. Configure Embedding model
embedding_model = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')

# Lets create the mainpage

st.title(':green[CHATBOT : ] :grey[AI Assisted chatbot using RAG]')
tips='''
Follow the steps to use the application :-
* Upload your PDF document in sidebar
* Write a query and start the chat
'''

st.text(tips)

# Lets create the sidebar

# Let's create the sidebar
st.sidebar.title(':grey[Upload your document]')
st.sidebar.subheader(':red[Supported format: PDF only]')
pdf_file = st.sidebar.file_uploader('Upload here', type = ['pdf'])

if pdf_file:
    st.sidebar.success('PDF uploaded successfully')

    file_text = text_extractor(pdf_file)

    # STEP 1 Chunking

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    chunks = splitter.split_text(file_text)


    # STEP 2

    vector_store = FAISS.from_texts(chunks,embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={'k':3})

    def generate_content(query):
        # STEP 3  Retriever (R)

        retrieved_docs = retriever.invoke(query)
        context ='\n'.join([d.page_content for d in retrieved_docs])

        # STEP 4 Augmenting (A)

        augmented_prompt = f'''
        <Role> You are a helpful assistant using RAG.
        <Goal> Answer the question asked by the user. Here is the question{query}
        <Context> Here are the documents retrieved from vector database to support the answer which you have to generate {context}
        '''

        # STEP 5 Generate (G)

        response = model.generate_content(augmented_prompt)

        return response.text
    
    # Create Chatbot in order to start the conversation

    # To initialize a chat create history if not created

    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display the history
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.info(f":blue[USER:] {msg['text']}")
        else:
            st.warning(f":red[CHATBOT:] {msg['text']}") 

    # Input from the user using streamlit form
    with st.form('CHATBOT FORM',clear_on_submit=True):
        user_query = st.text_area(':grey[ASK ANYTHING]')
        send = st.form_submit_button('SEND')

    # Start the conversation and append output and query in history
    if user_query and send:
        st.session_state.history.append({
        'role': 'user',
        'text': user_query.strip()
        })
        st.session_state.history.append({'role':'chatbot','text':generate_content(user_query)})
        st.rerun()










    





