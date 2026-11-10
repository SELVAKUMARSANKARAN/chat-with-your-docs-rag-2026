"""Streamlit UI for Chat With Your Docs."""
import hashlib
import logging
import os
import re

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from rag import extract_document, split_chunks, embed_texts, DocumentIndex, answer_question

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

st.set_page_config(page_title='Chat With Your Docs', page_icon='📚', layout='wide')
st.title('📚 Chat With Your Docs')
st.caption('Ask questions grounded in your PDF and TXT documents.')

chat_model = os.getenv('CHAT_MODEL', 'llama3.2:3b')
embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')


def get_conversational_reply(message: str) -> str | None:
    normalized = re.sub(r'[^\w\s]', '', message.strip().lower())
    normalized = ' '.join(normalized.split())
    replies = {
        'hi': 'Hello! What would you like to know about your documents?',
        'hello': 'Hello! What would you like to know about your documents?',
        'hey': 'Hello! How can I help with your documents?',
        'thanks': "You're welcome! Feel free to ask another question.",
        'thank you': "You're welcome! Feel free to ask another question.",
        'thank you so much': "You're very welcome!",
        'ok': 'Sure! Let me know if you have another question.',
        'okay': 'Sure! Let me know if you have another question.',
        'ok thanks': "You're welcome! Feel free to ask another question.",
        'okay thanks': "You're welcome! Feel free to ask another question.",
        'ok thank you': "You're welcome! Feel free to ask another question.",
        'okay thank you': "You're welcome! Feel free to ask another question.",
        'great': 'Glad I could help! Feel free to ask another question.',
        'great thanks': "You're welcome! Happy to help.",
        'good': 'Glad I could help!',
        'nice': 'Glad you found it helpful!',
        'perfect': 'Great! Let me know if you have another question.',
        'awesome': 'Happy to help!',
        'got it': 'Great! Feel free to ask another question.',
        'understood': 'Let me know if you have any other questions.',
        'cool': 'Glad I could help!',
        'alright': 'Sure! Let me know if you have another question.',
        'sounds good': 'Great! Let me know if you have another question.',
        'very good': 'Glad I could help!',
        'well done': 'Thank you! Feel free to ask another question.',
        'bye': 'Goodbye! Have a great day!',
    }
    return replies.get(normalized)


@st.cache_resource(show_spinner='Loading local embedding model (first run downloads files)...')
def load_model(name):
    return SentenceTransformer(name)


if 'index' not in st.session_state:
    st.session_state.index = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'fingerprint' not in st.session_state:
    st.session_state.fingerprint = None

with st.sidebar:
    st.header('Your documents')
    uploads = st.file_uploader('Choose PDF or TXT files', type=['pdf', 'txt'], accept_multiple_files=True)
    st.caption('Documents are held in this session. The first run downloads the embedding model.')
    build = st.button('Process documents', type='primary', use_container_width=True)
    if st.button('Clear chat', use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.session_state.index is not None:
        st.success(f'Ready: {len(st.session_state.index.chunks)} chunks indexed')

if build:
    if not uploads:
        st.warning('Upload at least one document.')
    else:
        try:
            documents = []
            fingerprints = []
            with st.spinner('Extracting text, chunking and generating embeddings...'):
                for upload in uploads:
                    data = upload.getvalue()
                    if len(data) > 15 * 1024 * 1024:
                        raise ValueError(f'{upload.name} exceeds the 15 MB per-file limit.')
                    fingerprints.append((upload.name, hashlib.sha256(data).hexdigest()))
                    documents.extend(extract_document(upload.name, data))
                chunks = split_chunks(documents)
                if not chunks:
                    raise ValueError('No extractable text found. Scanned PDFs need OCR, which this MVP does not support.')
                if len(chunks) > 1000:
                    raise ValueError('Too many chunks for this demo (maximum 1000).')
                vectors = embed_texts(load_model(embedding_model), [chunk.text for chunk in chunks])
                st.session_state.index = DocumentIndex(chunks, vectors)
                st.session_state.fingerprint = fingerprints
                st.session_state.messages = []
            st.success(f'Processed {len(uploads)} files into {len(chunks)} chunks.')
        except Exception as exc:
            logging.exception('Document processing failed')
            st.error(f'Could not process documents: {exc}')

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

question = st.chat_input('Ask something about your documents...', disabled=st.session_state.index is None)
if question:
    prior = [{'role': m['role'], 'content': m['content']} for m in st.session_state.messages]
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        conversational_reply = get_conversational_reply(question)
        if conversational_reply is not None:
            st.markdown(conversational_reply)
            st.session_state.messages.append({'role': 'assistant', 'content': conversational_reply})
        else:
            try:
                with st.spinner('Searching documents and preparing an answer...'):
                    answer, _matches = answer_question(
                        load_model(embedding_model), st.session_state.index,
                        question, chat_model, prior
                    )
                st.markdown(answer)
                st.session_state.messages.append({'role': 'assistant', 'content': answer})
            except Exception as exc:
                logging.exception('Question answering failed')
                st.error(f'Unable to answer: {exc}')
