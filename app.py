import streamlit as st
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub


from qdrant_client import QdrantClient

st.set_page_config(page_title="AI usecase evaluation", layout="centered")
st.title("AI usecase evaluation")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("memory", ConversationBufferMemory(
    memory_key="history",
    input_key="input",
    output_key="output",
    return_messages=True
))

# Connect to local qdrant instance
qdrant_client = QdrantClient("localhost:6333")
embeddings = OllamaEmbeddings(model="llama3.2:latest")

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="demo_collection",
    embedding=embeddings,
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = hub.pull("rlm/rag-prompt")

llm = ChatOllama(model="llama3.2:latest", streaming=True)
embeddings = OllamaEmbeddings(model="llama3")

# RAG chain
rag_chain = (
    {
        "context": vector_store.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

def stringify_history(history):
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in history if isinstance(msg, dict))


for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

if prompt := st.chat_input("Your prompt here"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    history = st.session_state.memory.load_memory_variables({"input": prompt})["history"]
    history.append({"role": "user", "content": prompt})

    flat_history = history[0] if isinstance(history, list) and isinstance(history[0], list) else history
    formatted_history = stringify_history(flat_history)

    with st.chat_message("assistant"):
        response = rag_chain.invoke(formatted_history)
        st.markdown(response)  # display in chat
        assistant_content = response  # store for memory

    st.session_state.memory.save_context({"input": prompt}, {"output": assistant_content})
    st.session_state.messages.append({"role": "assistant", "content": assistant_content})
