import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.messages import convert_to_messages, HumanMessage, AIMessage
from qdrant_client import QdrantClient
from langchain_core.retrievers import BaseRetriever
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(page_title="AI Usecase Evaluation", layout="centered")
st.title("AI Usecase Evaluation in Finance")

# Initialize session state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("current_category", 0)  # 0=initial, 1=benefits, 2=pressure, 3=readiness, 4=complete ; mapping in CATEGORY_NAMES
st.session_state.setdefault("category_info", {
    "perceived_benefits": "",
    "external_pressure": "",
    "organizational_readiness": ""
})
st.session_state.setdefault("use_case_overview", "")
st.session_state.setdefault("evaluation_complete", False)
st.session_state.setdefault("category_evaluations", {})

CATEGORY_NAMES = {
    1: "perceived_benefits",
    2: "external_pressure", 
    3: "organizational_readiness"
}

# LLM and embeddings setup
llm = ChatOpenAI(
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.1
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Connect to local Qdrant instance for knowledge base
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="banking_ai_usecases_small",
    embedding=embeddings
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
    
# ORCHESTRATOR FUNCTION
def process_user_input(user_input):
    """Simple router based on current_category"""
    current_cat = st.session_state.current_category
    
    if current_cat == 0:
        # Store overview, move to 1
        st.session_state.use_case_overview = user_input
        st.session_state.current_category = 1
        return ask_for_category_info("perceived_benefits")
    
    elif current_cat == 1:
        return process_perceived_benefits_input(user_input)
    
    elif current_cat == 2:
        return process_external_pressure_input(user_input)
    
    elif current_cat == 3:
        return process_organizational_readiness_input(user_input)
    
    elif current_cat == 4:
        # Perform final evaluation
        return perform_final_evaluation()
    
    else:
        return "Evaluation already completed. Start a new conversation for a fresh evaluation."

# gets called to return json generated in sufficiency checker 
def call_llm_and_parse_json(prompt):
    """Call LLM with prompt and parse JSON response"""
    try:
        response = llm.invoke(prompt)
        content = response.content
        
        # Extract JSON from response
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
        else:
            json_str = content
            
        return json.loads(json_str)
    except Exception as e:
        # Fallback if parsing fails
        return {
            "is_sufficient": True,  # Proceed anyway
            "missing_aspects": [],
            "follow_up_question": ""
        }
    


def ask_for_category_info(category_name):
    """Generate initial question for specific category"""
    questions = {
        "perceived_benefits": "📈 **Now let's explore the Perceived Benefits of your AI use case.**\n\nPlease tell me about expected cost savings, revenue impact, customer experience improvements, and strategic alignment.",
        
        "external_pressure": "🔍 **Let's discuss External Pressure factors.**\n\nPlease provide information about regulatory requirements, competitive pressure, customer demands, and risk management considerations.",
        
        "organizational_readiness": "🏢 **Finally, let's assess your Organizational Readiness.**\n\nPlease share details about data quality, team skills, IT infrastructure, and change management readiness."
    }
    return questions.get(category_name, "Please provide more information.")


def evaluate_perceived_benefits(collected_info):
    """Evaluate the perceived benefits category and return structured assessment"""
    
    # Get relevant examples for context
    kb_examples = get_relevant_examples(f"perceived benefits AI use case finance {collected_info}")
    
    evaluation_prompt = f"""
    You are evaluating the PERCEIVED BENEFITS of an AI use case in finance.
    
    CONTEXT AND EXAMPLES:
    {kb_examples}
    
    COLLECTED INFORMATION:
    {collected_info}
    
    Provide a structured evaluation of perceived benefits covering:
    1. **Cost Reduction Analysis** (quantify savings potential)
    2. **Revenue Impact Assessment** (growth opportunities)
    3. **Customer Experience Improvements** (service enhancements)
    4. **Strategic Alignment** (business goal alignment)
    5. **Competitive Advantage** (market positioning)
    6. **Benefits Score** (1-100 based on strength of benefits)
    
    Return your analysis in clear markdown format.
    """
    
    try:
        response = llm.invoke(evaluation_prompt)
        return response.content
    except Exception as e:
        return f"Unable to evaluate perceived benefits: {str(e)}"


def evaluate_external_pressure(collected_info):
    """Evaluate the external pressure category and return structured assessment"""
    
    # Get relevant examples for context
    kb_examples = get_relevant_examples(f"external pressure regulatory compliance AI finance {collected_info}")
    
    evaluation_prompt = f"""
    You are evaluating the EXTERNAL PRESSURE factors of an AI use case in finance.
    
    CONTEXT AND EXAMPLES:
    {kb_examples}
    
    COLLECTED INFORMATION:
    {collected_info}
    
    Provide a structured evaluation of external pressures covering:
    1. **Regulatory Compliance** (mandatory requirements)
    2. **Competitive Pressure** (market forces)
    3. **Customer Demands** (service expectations)
    4. **Risk Management** (mitigation needs)
    5. **Industry Standards** (best practices)
    6. **Pressure Score** (1-100 based on urgency and impact)
    
    Return your analysis in clear markdown format.
    """
    
    try:
        response = llm.invoke(evaluation_prompt)
        return response.content
    except Exception as e:
        return f"Unable to evaluate external pressure: {str(e)}"


def evaluate_organizational_readiness(collected_info):
    """Evaluate the organizational readiness category and return structured assessment"""
    
    # Get relevant examples for context
    kb_examples = get_relevant_examples(f"organizational readiness AI implementation finance {collected_info}")
    
    evaluation_prompt = f"""
    You are evaluating the ORGANIZATIONAL READINESS for an AI use case in finance.
    
    CONTEXT AND EXAMPLES:
    {kb_examples}
    
    COLLECTED INFORMATION:
    {collected_info}
    
    Provide a structured evaluation of organizational readiness covering:
    1. **Data Quality & Availability** (data infrastructure)
    2. **Team Skills & Expertise** (human resources)
    3. **IT Infrastructure** (technical capabilities)
    4. **Change Management** (adoption readiness)
    5. **Budget & Resources** (financial commitment)
    6. **Readiness Score** (1-100 based on implementation feasibility)
    
    Return your analysis in clear markdown format.
    """
    
    try:
        response = llm.invoke(evaluation_prompt)
        return response.content
    except Exception as e:
        return f"Unable to evaluate organizational readiness: {str(e)}"



def process_perceived_benefits_input(user_input):
    """Process input for perceived benefits category"""
    st.session_state.category_info["perceived_benefits"] += f" {user_input}"
    
    sufficiency = check_perceived_benefits_sufficiency(
        st.session_state.category_info["perceived_benefits"]
    )
    
    if sufficiency["is_sufficient"]:
        # Evaluate this category and store the result
        evaluation = evaluate_perceived_benefits(st.session_state.category_info["perceived_benefits"])
        st.session_state.category_evaluations["perceived_benefits"] = evaluation
        
        st.session_state.current_category = 2
        return ask_for_category_info("external_pressure")
    else:
        missing = ', '.join(sufficiency['missing_aspects'])
        return f"I need more information about **Perceived Benefits**.\n\n**Missing:** {missing}\n\n{sufficiency['follow_up_question']}"

def process_external_pressure_input(user_input):
    """Process input for external pressure category"""
    st.session_state.category_info["external_pressure"] += f" {user_input}"
    
    sufficiency = check_external_pressure_sufficiency(
        st.session_state.category_info["external_pressure"]
    )
    
    if sufficiency["is_sufficient"]:
        # Evaluate this category and store the result
        evaluation = evaluate_external_pressure(st.session_state.category_info["external_pressure"])
        st.session_state.category_evaluations["external_pressure"] = evaluation
        
        st.session_state.current_category = 3
        return ask_for_category_info("organizational_readiness")
    else:
        missing = ', '.join(sufficiency['missing_aspects'])
        return f"I need more information about **External Pressure**.\n\n**Missing:** {missing}\n\n{sufficiency['follow_up_question']}"

def process_organizational_readiness_input(user_input):
    """Process input for organizational readiness category"""
    st.session_state.category_info["organizational_readiness"] += f" {user_input}"
    
    sufficiency = check_organizational_readiness_sufficiency(
        st.session_state.category_info["organizational_readiness"]
    )
    
    if sufficiency["is_sufficient"]:
        # Evaluate this category and store the result
        evaluation = evaluate_organizational_readiness(st.session_state.category_info["organizational_readiness"])
        st.session_state.category_evaluations["organizational_readiness"] = evaluation
        
        st.session_state.current_category = 4
        return perform_final_evaluation()
    else:
        missing = ', '.join(sufficiency['missing_aspects'])
        return f"I need more information about **Organizational Readiness**.\n\n**Missing:** {missing}\n\n{sufficiency['follow_up_question']}"

# TODO: Define required information for each category
def check_perceived_benefits_sufficiency(collected_info):
    required_information = """
    REQUIRED INFORMATION FOR PERCEIVED BENEFITS:
    - Cost reduction percentage or amount (required)
    - Revenue impact estimation (required)
    - Customer experience metrics (optional but preferred)
    - Strategic alignment with business goals (required)
    - Competitive advantage description (optional)
    """
    
    # specific prompting for this exact category
    prompt = f"""
    You are evaluating ONLY the perceived benefits of an AI use case.
    Focus specifically on the following requirements:

    {required_information}
    
    Information provided: {collected_info}
    
    Return JSON: {{"is_sufficient": true/false, "missing_aspects": [...], "follow_up_question": "..."}}
    """

    return call_llm_and_parse_json(prompt)


def check_external_pressure_sufficiency(collected_info):
    required_information = """
    REQUIRED INFORMATION FOR EXTERNAL PRESSURE:
    - Regulatory or compliance requirements (required)
    - Competitive market pressure (required)
    - Customer demands and expectations (optional but preferred)
    - Risk management considerations (required)
    - Industry trends and standards (optional)
    """
    
    prompt = f"""
    You are evaluating ONLY the external pressure factors of an AI use case.
    Focus specifically on the following requirements:

    {required_information}
    
    Information provided: {collected_info}
    
    Return JSON: {{"is_sufficient": true/false, "missing_aspects": [...], "follow_up_question": "..."}}
    """
    
    return call_llm_and_parse_json(prompt)


def check_organizational_readiness_sufficiency(collected_info):
    required_information = """
    REQUIRED INFORMATION FOR ORGANIZATIONAL READINESS:
    - Data quality and availability (required)
    - Team skills and expertise (required)
    - IT infrastructure capabilities (required)
    - Change management readiness (optional but preferred)
    - Budget and resource allocation (optional)
    """
    
    prompt = f"""
    You are evaluating ONLY the organizational readiness of an AI use case.
    Focus specifically on the following requirements:

    {required_information}
    
    Information provided: {collected_info}
    
    Return JSON: {{"is_sufficient": true/false, "missing_aspects": [...], "follow_up_question": "..."}}
    """
    
    return call_llm_and_parse_json(prompt)

# TODO: evaluate each category when switching to the next and store it for final evaluation
def perform_final_evaluation():
    """Perform the final evaluation using all category evaluations"""
    
    # Get all category evaluations
    category_evaluations = st.session_state.get("category_evaluations", {})
    
    # Collect all information for context
    full_context = f"""
    Use Case Overview: {st.session_state.use_case_overview}
    
    CATEGORY EVALUATIONS:
    
    Perceived Benefits Evaluation:
    {category_evaluations.get('perceived_benefits', 'Not evaluated')}
    
    External Pressure Evaluation:
    {category_evaluations.get('external_pressure', 'Not evaluated')}
    
    Organizational Readiness Evaluation:
    {category_evaluations.get('organizational_readiness', 'Not evaluated')}
    """
    
    # Get relevant examples from knowledge base
    kb_examples = get_relevant_examples(st.session_state.use_case_overview)

    # Create final evaluation prompt
    evaluation_prompt = f"""
    You are an AI expert providing a FINAL COMPREHENSIVE evaluation of an AI use case in finance.
    
    CONTEXT AND EXAMPLES:
    {kb_examples}
    
    DETAILED ANALYSIS FROM CATEGORIES:
    {full_context}
    
    Based on the detailed category evaluations above, provide a final assessment covering:
    
    1. **Overall Viability Score** (1-100, weighted average of category scores)
    2. **Executive Summary** (key findings and recommendation)
    3. **Category Synopsis** (brief summary of each category's key points)
    4. **Implementation Priority** (High/Medium/Low with rationale)
    5. **Key Success Factors** (3-5 critical elements for success)
    6. **Major Risk Factors** (3-5 main concerns to address)
    7. **Next Steps** (specific recommended actions)
    
    Format your response in clear markdown with sections and use the insights from the detailed category evaluations.
    """
    
    try:
        response = llm.invoke(evaluation_prompt)
        st.session_state.evaluation_complete = True
        return response.content
    except Exception as e:
        return f"Unable to complete final evaluation: {str(e)}"


# Display chat messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Chat input
if prompt := st.chat_input("Tell me about your AI use case..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process with category-based workflow
    with st.chat_message("assistant"):
        response = process_user_input(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


# Add a section to explain how the app works
with st.expander("How this works"):
    st.markdown("""
    This AI use case evaluation system uses a category-by-category approach:
    
    1. **Step-by-Step Evaluation**:
       - **Perceived Benefits**: Evaluates expected advantages and ROI
       - **External Pressure**: Assesses regulatory and competitive factors  
       - **Organizational Readiness**: Analyzes implementation capabilities

    2. **Information Gathering**: The system asks follow-up questions until sufficient information is collected for each category.

    3. **Knowledge Base**: Searches a curated database of AI use cases in finance for relevant examples.
    
    This ensures thorough evaluation based on established examples and systematic analysis.
    """)
