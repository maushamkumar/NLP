import streamlit as st
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import time

# --- Configuration ---
GPT4ALL_MODEL_PATH = "./models/orca-mini-3b-gguf2-q4_0.gguf"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

SAMPLE_DOCUMENT_TEXT = """The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France...
[Full document content as in original code]"""

@st.cache_resource
def load_embeddings_model(model_name):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

@st.cache_resource
def load_llm(model_path):
    if not os.path.exists(model_path):
        st.error(f"GPT4All model not found at {model_path}.")
        return None
    try:
        llm = GPT4All(model=model_path, backend="gptj", verbose=True, streaming=True)
        return llm
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        return None

@st.cache_resource(show_spinner="Processing document and building vector store...")
def create_vector_store(_docs, _embeddings):
    documents = [Document(page_content=_docs, metadata={"source": "sample_document"})]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(documents)
    try:
        vectorstore = FAISS.from_documents(doc_chunks, _embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating FAISS vector store: {e}")
        return None

# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(page_title="RAG Q&A Assistant with Memory")
    st.title("🧠 RAG Q&A Assistant with Chat History")

    model_path_input = st.sidebar.text_input("Path to GPT4All Model File (.gguf)", GPT4ALL_MODEL_PATH)
    _embeddings = load_embeddings_model(EMBEDDING_MODEL_NAME)
    llm = load_llm(model_path_input)
    vectorstore = create_vector_store(SAMPLE_DOCUMENT_TEXT, _embeddings)

    if not llm or not vectorstore:
        return

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,
        output_key="answer"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Enter your question:")

    if st.button("Submit"):
        if user_query:
            with st.spinner("Thinking..."):
                response = conv_chain({"question": user_query})
                answer = response.get("answer", "No answer found.")
                st.session_state.chat_history.append((user_query, answer))

                st.subheader("✅ Answer:")
                st.markdown(answer)

                source_documents = response.get("source_documents", [])
                if source_documents:
                    st.subheader("🔍 Retrieved Context Snippets:")
                    for i, doc in enumerate(source_documents):
                        with st.expander(f"Snippet {i+1} (Source: {doc.metadata.get('source', 'N/A')})"):
                            st.markdown(doc.page_content)

    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("🗂️ Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")

    st.markdown("---")
    with st.expander("📄 Sample Document Content"):
        st.text(SAMPLE_DOCUMENT_TEXT)

if __name__ == "__main__":
    if not os.path.exists("./models"):
        os.makedirs("./models")
    main()
