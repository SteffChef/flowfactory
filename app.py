import streamlit as st
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import convert_to_messages, HumanMessage, AIMessage
from langgraph.types import interrupt
from qdrant_client import QdrantClient
from langchain_core.retrievers import BaseRetriever

# Page configuration
st.set_page_config(page_title="AI Usecase Evaluation", layout="centered")
st.title("AI Usecase Evaluation in Finance")

# Initialize session state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("memory", ConversationBufferMemory(
    memory_key="history",
    input_key="input",
    output_key="output",
    return_messages=True
))
st.session_state.setdefault("waiting_for_human_input", False)
st.session_state.setdefault("current_query", "")
st.session_state.setdefault("agent_conversation", [])

# LLM and embeddings setup
llm = ChatOllama(model="llama3.2:latest", streaming=True)
embeddings = OllamaEmbeddings(model="llama3.2:latest")

# Connect to local Qdrant instance for knowledge base
qdrant_client = QdrantClient("localhost:6333")
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="demo_collection",
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# RAG Chain for retrieving relevant AI use cases
rag_prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Function to get relevant examples from knowledge base
def get_relevant_examples(query: str) -> str:
    """Retrieve relevant AI use case examples from the knowledge base."""
    try:
        response = rag_chain.invoke(query)
        return response
    except Exception as e:
        return f"Unable to retrieve examples: {str(e)}"

# Function to collect human input when an agent requests it
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    st.session_state.waiting_for_human_input = True
    st.session_state.current_query = query
    
    # First try to get information from our knowledge base
    kb_response = get_relevant_examples(query)
    
    # Return knowledge base response or default if not useful
    if kb_response and len(kb_response) > 20:
        return f"Based on our knowledge base: {kb_response}"
    return "The information is not available at this moment. Please continue with what you know."

# Tool to access the RAG system for the agents
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant AI use case examples in finance."""
    examples = get_relevant_examples(query)
    return f"Knowledge base examples: {examples}"

# Create the agents with access to both human assistance and knowledge base
perceived_benefits_agent = create_react_agent(
    model=llm,
    tools=[human_assistance, search_knowledge_base],
    prompt=(
        "You are a perceived benefits agent for evaluating AI usecases in finance.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with evaluating AI usecases in terms of perceived benefits\n"
        "- Focus on potential ROI, efficiency gains, accuracy improvements, and competitive advantages\n"
        "- Use the search_knowledge_base tool to find relevant examples before asking for human help\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="perceived_benefits_agent",
)

external_pressure_agent = create_react_agent(
    model=llm,
    tools=[human_assistance, search_knowledge_base],
    prompt=(
        "You are an external pressure agent for evaluating AI usecases in finance.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with evaluating AI usecases in terms of external pressure\n"
        "- Focus on regulatory requirements, competitor actions, customer expectations, and market trends\n"
        "- Use the search_knowledge_base tool to find relevant examples before asking for human help\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="external_pressure_agent",
)

organizational_readiness_agent = create_react_agent(
    model=llm,
    tools=[human_assistance, search_knowledge_base],
    prompt=(
        "You are an organizational readiness agent for evaluating AI usecases in finance.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with evaluating AI usecases in terms of organizational readiness\n"
        "- Focus on technical capabilities, data quality, staff skills, governance frameworks, and implementation feasibility\n"
        "- Use the search_knowledge_base tool to find relevant examples before asking for human help\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="organizational_readiness_agent",
)

# Create the supervisor with access to the knowledge base
supervisor = create_supervisor(
    model=llm,
    agents=[perceived_benefits_agent, external_pressure_agent, organizational_readiness_agent],
    tools=[search_knowledge_base],
    prompt=(
        "You are a supervisor managing three agents for evaluating AI use cases in finance:\n"
        "- perceived_benefits_agent: Assign tasks to this agent regarding perceived benefits of AI use cases.\n"
        "- external_pressure_agent: Assign tasks to this agent regarding external pressures on AI use cases.\n"
        "- organizational_readiness_agent: Assign tasks to this agent regarding organizational readiness for AI use cases.\n\n"
        "Before assigning tasks, use the search_knowledge_base tool to gather relevant examples.\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself.\n"
        "When responding to the user, synthesize all findings from the agents into a comprehensive assessment."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

# Function to collect agent responses
class ConversationCollector(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.conversation = []
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.conversation.append(token)
        
    def get_conversation(self):
        return "".join(self.conversation)

# Display chat history
for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

# Display human assistance prompt if needed
if st.session_state.waiting_for_human_input:
    with st.container():
        st.info(f"Agent needs information: {st.session_state.current_query}")
        human_response = st.text_input("Your response to the agent:", key="human_assist_input")
        if st.button("Submit Response"):
            st.session_state.waiting_for_human_input = False
            # In a real implementation, we'd need to feed this back to the agent
            # This is a placeholder for now

# Handle new user input
if prompt := st.chat_input("Ask about AI use cases in finance..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process with the agent system
    with st.chat_message("assistant"):
        collector = ConversationCollector()
        callbacks = [collector]
        
        # First get relevant information from the knowledge base
        with st.spinner("Retrieving relevant examples..."):
            kb_examples = get_relevant_examples(prompt)
            
        with st.spinner("Agents are evaluating your use case..."):
            # Run the supervisor without callbacks parameter
            for chunk in supervisor.stream(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": f"Relevant examples from the knowledge base: {kb_examples}"
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]
                }
            ):
                # This would update the conversation in real-time if supported
                pass
            
            # Extract the final response from the supervisor
            if "supervisor" in chunk:
                messages = chunk["supervisor"]["messages"]
                # Get the last assistant message
                response = "No response generated"
                for msg in reversed(messages):
                    # Use proper attribute access for Message objects
                    if hasattr(msg, 'type') and msg.type == 'ai':
                        response = msg.content
                        break
                    # For dictionary messages
                    elif isinstance(msg, dict) and msg.get("role") == "assistant":
                        response = msg.get("content", "No response generated")
                        break
            else:
                response = "The agents couldn't process your request."
            
            # Display the response
            st.markdown(response)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.memory.save_context({"input": prompt}, {"output": response})

# Add a section to explain how the app works
with st.expander("How this works"):
    st.markdown("""
    This AI use case evaluation system combines retrieval-augmented generation (RAG) with a multi-agent architecture:
    
    1. **Knowledge Base**: The system first searches a curated database of AI use cases in finance to find relevant examples.
    
    2. **Specialized Agents**:
       - **Perceived Benefits Agent**: Evaluates potential advantages of implementing AI solutions
       - **External Pressure Agent**: Assesses market forces, regulations, and competitive factors
       - **Organizational Readiness Agent**: Analyzes your organization's capability to implement AI
    
    3. **Supervisor**: Coordinates the agents and synthesizes their findings into a comprehensive assessment.
    
    This approach ensures you get a thorough evaluation based on both established examples and expert analysis.
    """)
