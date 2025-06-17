import streamlit as st
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from qdrant_client import QdrantClient
import json
import os

# Page configuration
st.set_page_config(page_title="AI Usecase Evaluation in Finance", layout="centered")
st.title("AI Usecase Evaluation in Finance")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Track information request attempts  
if "info_requests" not in st.session_state:
    st.session_state.info_requests = 0

# Track accumulated user input to prevent forgetting
if "accumulated_input" not in st.session_state:
    st.session_state.accumulated_input = ""
    
st.session_state.setdefault("category_scores", {
    "perceived_benefits": None,
    "external_pressure": None, 
    "organizational_readiness": None
})
st.session_state.setdefault("evaluation_details", {
    "perceived_benefits": "",
    "external_pressure": "",
    "organizational_readiness": ""
})

# Constants
MAX_INFO_REQUESTS = 2  # Maximum number of times to ask for more information

# Read evaluation criteria
evaluation_criteria_path = os.path.join(os.path.dirname(__file__), "evaluation_criteria.txt")
with open(evaluation_criteria_path, "r") as file:
    evaluation_criteria = file.read()

# Extract criteria for each category
perceived_benefits_criteria = evaluation_criteria.split("(1) Perceived Benefits")[1].split("(2) External Pressure")[0].strip()
external_pressure_criteria = evaluation_criteria.split("(2) External Pressure")[1].strip() if "(3) Organizational Readiness" not in evaluation_criteria else evaluation_criteria.split("(2) External Pressure")[1].split("(3) Organizational Readiness")[0].strip()
organizational_readiness_criteria = evaluation_criteria.split("(3) Organizational Readiness")[1].strip() if "(3) Organizational Readiness" in evaluation_criteria else ""

# LLM setup
llm = ChatOllama(model="llama3.2:latest", temperature=0.1, streaming=True)

# Setup RAG if available
try:
    embeddings = OllamaEmbeddings(model="llama3.2:latest")
    qdrant_client = QdrantClient("localhost:6333")
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="demo_collection",
        embedding=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    rag_available = True
except Exception as e:
    st.warning("Knowledge base not available. Continuing without RAG capabilities.")
    retriever = None
    rag_available = False

# Function to get relevant examples from knowledge base
def get_relevant_examples(query):
    if not rag_available or not retriever:
        return ""
    
    try:
        rag_prompt = hub.pull("rlm/rag-prompt")
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain.invoke(query)
    except:
        return ""

# Function to safely extract JSON from text
def extract_json_from_text(text):
    if not text:
        return None
    try:
        # Find JSON content between curly braces
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            return json.loads(json_str)
        return None
    except:
        return None

# Function to evaluate a single category
def evaluate_category(category, criteria, prompt, examples=""):
    try:
        # Prepare evaluation prompt
        evaluation_prompt = f"""
        You are an AI expert evaluating {category} of an AI use case in finance.
        
        EVALUATION CRITERIA:
        {criteria}
        
        RELEVANT EXAMPLES AND KNOWLEDGE:
        {examples}
        
        USER QUERY:
        {prompt}
        
        Please provide a comprehensive evaluation of this AI use case focusing on {category}.
        
        Score this category from 1-100 based on the following scale:
        - 1-20: Very poor/High risk
        - 21-40: Poor/Significant concerns
        - 41-60: Average/Moderate potential
        - 61-80: Good/Strong potential
        - 81-100: Excellent/Exceptional potential
        
        Respond with a JSON object with the following structure:
        {{
            "score": <1-100>,
            "evaluation": "<detailed explanation of your assessment>",
            "key_strengths": ["<strength1>", "<strength2>", ...],
            "key_weaknesses": ["<weakness1>", "<weakness2>", ...],
            "recommendations": ["<recommendation1>", "<recommendation2>", ...]
        }}
        
        Make sure the JSON is properly formatted with no errors.
        """
        
        # Get evaluation
        response = llm.invoke(evaluation_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON
        result = extract_json_from_text(content)
        if result and "score" in result:
            return result
        
        # If extraction failed, try a simplified retry
        retry_prompt = f"""
        Please evaluate the {category} of the following AI use case in finance:
        "{prompt}"
        
        Respond with a simple JSON:
        {{
            "score": <number between 1-100>,
            "evaluation": "<brief assessment>",
            "key_strengths": ["<strength1>"],
            "key_weaknesses": ["<weakness1>"],
            "recommendations": ["<recommendation1>"]
        }}
        """
        response = llm.invoke(retry_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        result = extract_json_from_text(content)
        
        if result and "score" in result:
            return result
            
        # Last resort - create a simple result
        return {
            "score": 50,
            "evaluation": f"Unable to generate a complete assessment for {category}.",
            "key_strengths": ["Not determined"],
            "key_weaknesses": ["Not determined"],
            "recommendations": ["Provide more specific information"]
        }
    except Exception as e:
        st.error(f"Error evaluating {category}: {str(e)}")
        return {
            "score": 50,
            "evaluation": f"Error during evaluation of {category}.",
            "key_strengths": ["Not determined"],
            "key_weaknesses": ["Not determined"],
            "recommendations": ["Provide more specific information"]
        }

# Function to check if we have enough information
def check_information_sufficiency(prompt):
    # If we've already asked for more information twice, proceed anyway
    if st.session_state.info_requests >= MAX_INFO_REQUESTS:
        return {"is_sufficient": True, "missing_information": [], "feedback": "Proceeding with evaluation based on available information."}
    
    try:
        check_prompt = f"""
        You're evaluating whether there's sufficient information about an AI use case in finance.
        
        USER QUERY: {prompt}
        
        Is there enough information to assess:
        1. Perceived Benefits (value proposition, efficiency gains, etc.)
        2. External Pressure (regulatory, market forces, etc.)
        3. Organizational Readiness (technical capabilities, skills, etc.)
        
        Be lenient in your assessment. If there's basic information about the AI use case, consider it sufficient.
        
        Respond true/false and explain why: {{
            "is_sufficient": true/false,
            "missing_information": ["specific missing info"],
            "feedback": "brief explanation"
        }}
        """
        
        response = llm.invoke(check_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        result = extract_json_from_text(content)
        
        if result and "is_sufficient" in result:
            return result
        return {"is_sufficient": True, "missing_information": [], "feedback": ""}
    except:
        # On error, proceed with evaluation
        return {"is_sufficient": True, "missing_information": [], "feedback": ""}

# Display chat history
for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

# Handle user input
if prompt := st.chat_input("Describe an AI use case in finance for evaluation..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Accumulate user input
    if st.session_state.accumulated_input:
        st.session_state.accumulated_input += f"\n\nAdditional information: {prompt}"
    else:
        st.session_state.accumulated_input = prompt
    
    # Process with evaluation system
    with st.chat_message("assistant"):
        # Use accumulated input for better context retention
        full_input = st.session_state.accumulated_input
        
        # Check if enough information
        with st.spinner("Analyzing your request..."):
            sufficiency = check_information_sufficiency(full_input)
            
        if not sufficiency.get("is_sufficient", True) and st.session_state.info_requests < MAX_INFO_REQUESTS:
            # Increment the information request counter
            st.session_state.info_requests += 1
            
            # Extract missing information
            missing_items = sufficiency.get("missing_information", [
                "Specific AI technology being used",
                "Financial use case details",
                "Organizational context and capabilities"
            ])
            
            # Format as bullet points with proper spacing
            missing_info = ""
            for item in missing_items:
                missing_info += f"- {item}\n"
            
            # Create a cleaner, more direct response
            response = f"**More Information Needed ({st.session_state.info_requests}/{MAX_INFO_REQUESTS})**\n\nTo evaluate your AI use case, I need these details:\n\n{missing_info}\n{sufficiency.get('feedback', '')}"
            
            # Display immediately
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Add suggestion for what to include
            st.info("Try including details about the AI technology, specific use case, expected benefits, and organizational context.")
            
        else:
            # Reset info request counter for next evaluation
            st.session_state.info_requests = 0
            
            # Get knowledge base examples if available
            kb_examples = ""
            if rag_available:
                with st.spinner("Retrieving relevant examples..."):
                    kb_examples = get_relevant_examples(full_input)
            
            # Reset scores
            st.session_state.category_scores = {
                "perceived_benefits": None,
                "external_pressure": None, 
                "organizational_readiness": None
            }
            
            st.session_state.evaluation_details = {
                "perceived_benefits": "",
                "external_pressure": "",
                "organizational_readiness": ""
            }
            
            # Evaluate each category
            with st.spinner("Evaluating perceived benefits..."):
                pb_result = evaluate_category("perceived benefits", perceived_benefits_criteria, full_input, kb_examples)
                st.session_state.category_scores["perceived_benefits"] = pb_result["score"]
                st.session_state.evaluation_details["perceived_benefits"] = pb_result["evaluation"]
            
            with st.spinner("Evaluating external pressure..."):
                ep_result = evaluate_category("external pressure", external_pressure_criteria, full_input, kb_examples)
                st.session_state.category_scores["external_pressure"] = ep_result["score"]
                st.session_state.evaluation_details["external_pressure"] = ep_result["evaluation"]
            
            with st.spinner("Evaluating organizational readiness..."):
                or_result = evaluate_category("organizational readiness", organizational_readiness_criteria, full_input, kb_examples)
                st.session_state.category_scores["organizational_readiness"] = or_result["score"]
                st.session_state.evaluation_details["organizational_readiness"] = or_result["evaluation"]
            
            # Calculate total score
            scores = st.session_state.category_scores
            total_score = (
                scores["perceived_benefits"] * 0.35 + 
                scores["external_pressure"] * 0.35 + 
                scores["organizational_readiness"] * 0.3
            )
            
            # Determine interpretation
            if total_score <= 20:
                interpretation = "Not feasible for implementation"
            elif total_score <= 40:
                interpretation = "High risk, substantial improvements needed"
            elif total_score <= 60:
                interpretation = "Moderate potential, specific improvements required"
            elif total_score <= 80:
                interpretation = "Good potential, minor improvements suggested"
            else:
                interpretation = "Excellent potential, ready for implementation"
            
            # Create strengths, weaknesses and recommendations sections
            strengths = []
            if pb_result.get("key_strengths"):
                strengths.extend([f"- {s}" for s in pb_result.get("key_strengths", [])[:2]])
            if ep_result.get("key_strengths"):
                strengths.extend([f"- {s}" for s in ep_result.get("key_strengths", [])[:2]])
            if or_result.get("key_strengths"):
                strengths.extend([f"- {s}" for s in or_result.get("key_strengths", [])[:2]])

            weaknesses = []
            if pb_result.get("key_weaknesses"):
                weaknesses.extend([f"- {s}" for s in pb_result.get("key_weaknesses", [])[:2]])
            if ep_result.get("key_weaknesses"):
                weaknesses.extend([f"- {s}" for s in ep_result.get("key_weaknesses", [])[:2]])
            if or_result.get("key_weaknesses"):
                weaknesses.extend([f"- {s}" for s in or_result.get("key_weaknesses", [])[:2]])

            recommendations = []
            if pb_result.get("recommendations"):
                recommendations.extend([f"- {s}" for s in pb_result.get("recommendations", [])[:2]])
            if ep_result.get("recommendations"):
                recommendations.extend([f"- {s}" for s in ep_result.get("recommendations", [])[:2]])
            if or_result.get("recommendations"):
                recommendations.extend([f"- {s}" for s in or_result.get("recommendations", [])[:2]])

            # Create summary using properly formatted lists
            response = f"# AI Use Case Evaluation Summary\n\n"
            response += f"## Executive Summary\n\n"
            response += f"I've evaluated your AI use case based on three key dimensions. "
            response += f"The overall score is **{total_score:.1f}/100** ({interpretation}).\n\n"

            response += f"## Category Scores\n\n"
            response += f"- **Perceived Benefits**: {scores['perceived_benefits']}/100\n"
            response += f"- **External Pressure**: {scores['external_pressure']}/100\n" 
            response += f"- **Organizational Readiness**: {scores['organizational_readiness']}/100\n\n"

            if strengths:
                response += "## Key Strengths\n\n"
                response += "\n".join(strengths) + "\n\n"

            if weaknesses:
                response += "## Key Weaknesses\n\n"
                response += "\n".join(weaknesses) + "\n\n"

            if recommendations:
                response += "## Recommendations\n\n"
                response += "\n".join(recommendations)
            
            # Display the response
            st.markdown(response)
            
            # Display score visualization
            st.subheader("Evaluation Scores")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Perceived Benefits", scores["perceived_benefits"])
                with st.expander("Details"):
                    st.write(st.session_state.evaluation_details["perceived_benefits"])
            
            with col2:
                st.metric("External Pressure", scores["external_pressure"])
                with st.expander("Details"):
                    st.write(st.session_state.evaluation_details["external_pressure"])
            
            with col3:
                st.metric("Organizational Readiness", scores["organizational_readiness"])
                with st.expander("Details"):
                    st.write(st.session_state.evaluation_details["organizational_readiness"])
            
            st.metric("Total Score", f"{total_score:.1f}/100")
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Reset accumulated input after successful evaluation
            st.session_state.accumulated_input = ""

# Add explanation of the framework
with st.expander("About the Evaluation Framework"):
    st.markdown("""
    ### AI Use Case Evaluation Framework

    This application evaluates AI use cases in finance across three key dimensions:

    **1. Perceived Benefits (35%)**
    - Cost reductions and efficiency gains
    - Revenue growth potential
    - Customer experience improvements
    - Strategic alignment with business goals

    **2. External Pressure (35%)**
    - Regulatory and compliance requirements
    - Model explainability and transparency
    - Risk management and audit capabilities
    - Data privacy and market factors

    **3. Organizational Readiness (30%)**
    - Data quality and availability
    - Workforce expertise and capabilities
    - IT infrastructure and integration
    - Organizational culture and change readiness
    
    The evaluations are transparent in their reasoning and scoring methodology to provide an explainable assessment.
    """)

# Add a button to reset evaluation
if st.button("Start New Evaluation"):
    st.session_state.info_requests = 0
    st.session_state.accumulated_input = ""
    st.session_state.category_scores = {
        "perceived_benefits": None,
        "external_pressure": None, 
        "organizational_readiness": None
    }
    st.session_state.evaluation_details = {
        "perceived_benefits": "",
        "external_pressure": "",
        "organizational_readiness": ""
    }
    st.rerun()
