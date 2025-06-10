"""
Cached Components for PDF RAG Application

This module contains all the cached component initialization functions
for the PDF RAG application, including LLM, embeddings, vector store,
and RAG agent creation.
"""

import os
from typing import TypedDict, Annotated, Sequence, List, Dict
import streamlit as sl

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
import re
from rank_bm25 import BM25Okapi
import numpy as np
    


@sl.cache_resource
def create_rag_agent(_llm, _retriever):
    """
    Create RAG agent once per session.
    
    Args:
        _llm: Language model instance
        _retriever: Document retriever instance
        
    Returns:
        Compiled RAG agent graph
    """
    print("🤖 Creating RAG agent...")
    
    @tool
    def retriever_tool(query: str) -> str:
        """This tool searches and returns the information from a pdf document."""
        docs = _retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the pdf document."
        
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Document {i+1}:\n{doc.page_content}")
        
        return "\n\n".join(results)
    
    tools = [retriever_tool]
    llm_with_tools = _llm.bind_tools(tools)
    
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
    
    def should_continue(state: AgentState):
        """Check if the last message contains tool calls."""
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0
    
    system_prompt = """
    You are an intelligent AI assistant who answers questions about the pdf document loaded into your knowledge base.
    Use the retriever tool available to answer questions. You can make multiple calls if needed.
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    Please always cite the specific parts of the documents you use in your answers (e.g. page, section). 
    """
    
    tools_dict = {tool.name: tool for tool in tools}
    
    def call_llm(state: AgentState) -> AgentState:
        """Function to call the LLM with the current state."""
        messages = list(state['messages'])
        messages = [SystemMessage(content=system_prompt)] + messages
        message = llm_with_tools.invoke(messages)
        return {'messages': [message]}
    
    def take_action(state: AgentState) -> AgentState:
        """Execute tool calls from the LLM's response."""
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            if not t['name'] in tools_dict:
                result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
            else:
                result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}
    
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {True: "retriever_agent", False: END}
    )
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")
    
    rag_agent = graph.compile()
    
    # Save the graph (only once)
    try:
        graph_image = rag_agent.get_graph().draw_mermaid_png()
        with open("rag_agent_graph.png", "wb") as f:
            f.write(graph_image)
        print("✅ RAG agent graph saved as 'rag_agent_graph.png'")
    except Exception as e:
        print(f"⚠️ Could not save RAG agent graph: {e}")
    
    print("✅ RAG agent created successfully!")
    return rag_agent


@sl.cache_resource
def create_rag_agent_v2(_llm, _retriever, _llm_evaluator):
    """
    Create RAG agent with evaluation step.
    
    Args:
        _llm: Language model instance for main responses
        _retriever: Document retriever instance
        _llm_evaluator: Language model instance for evaluation
        
    Returns:
        Compiled RAG agent graph with evaluation
    """
    print("🤖 Creating RAG agent...")
    
    @tool
    def retriever_tool(query: str) -> str:
        """This tool searches and returns the information from a pdf document."""
        docs = _retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the pdf document."
        
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Document {i+1}:\n{doc.page_content}")
        
        return "\n\n".join(results)
    
    tools = [retriever_tool]
    llm_with_tools = _llm.bind_tools(tools)
    
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        user_query: str  # Store original user query for evaluation
        evaluation_score: float  # Store evaluation score
    
    
    def should_continue(state: AgentState):
        """Check if the last message contains tool calls."""
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


    system_prompt = """
    You are an intelligent AI assistant who answers questions about the pdf document loaded into your knowledge base.
    Use the retriever tool available to answer questions. You can make multiple calls if needed.
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    Please always cite the specific parts of the documents you use in your answers (e.g. page, section). 
    """
    
    
    evaluation_prompt = """

    You are a top 1 percent expert evaluator tasked with assesing top notch scientists. 
    Be critical and demanding in your evaluation     
    You are a top 1 percent expert evaluator tasked with assesing top notch scientists.
     Be critical and demanding in your evaluation     
     as your reputation as a top expert is on the line and you might dissapear if this task is not done at your best capability.
    EVALUATION PROCESS:      
    
    1. EXTRACT MAIN QUESTIONS:     
    
    First, identify the core questions being asked in the user query (ignore supporting context).      
    
    2. EVALUATE EACH CRITERION (a score in the range of 0.00 to 0.20 each):      
    
    - Relevance: How well does the response address the user's question?           
    
    - Accuracy: Is the information provided correct and well-supported?          
    
    - Completeness: Does the response fully answer the question? Does it include examples?          
    - Clarity: Is the response clear and well-structured so as to be understood by a any professional     
     and it has zero ambiguity?          
     
     - Citation: Are there citations? are there of a good enough quality? are they relevant to the user's question?        
     
    3. APPLY STRICT STANDARDS:     
    - Be highly critical - most responses should score 0.20-0.60 after summing the 5 components.     
    - Only exceptional responses deserve 0.60+     
    - 0.85+ answers are 1 in a 100     
    - Common issues that reduce scores: vague language, missing examples, weak citations, incomplete answers      
    
    User Query: {user_query}     
    Answer: {ai_response}      
    
    THINK STEP BY STEP:      
    
    Step 1 - Main Questions: [Identify the core questions]      
    
    Step 2 - Relevance Score: [Evaluate and assign a score in the range 0.00 to 0.20]      
    
    Step 3 - Accuracy Score: [Evaluate and assign a score in the range 0.00 to 0.20]      
    
    Step 4 - Completeness Score: [Evaluate and assign a score in the range 0.00 to 0.20]      
    
    Step 5 - Clarity Score: [Evaluate and assign a score in the range 0.00 to 0.20]      
    
    Step 6 - Citations Score: [Evaluate and assign a score in the range 0.00 to 0.20]      
    
    Step 7 - Total Score: [Sum all scores = X.XX]

    Write the scores for each component with almost no text. Then write the sum of all (X.XX, e.g. 0.48).      

    IMPORTANT: After your analysis, you MUST end your response with exactly this format, indicating the sum of all criteria:
    <<<SCORE>>>X.XX<<<END>>>

    """
    
    tools_dict = {tool.name: tool for tool in tools}
    
    def call_llm(state: AgentState) -> AgentState:
        """Function to call the LLM with the current state."""
        messages = list(state['messages'])
        messages = [SystemMessage(content=system_prompt)] + messages
        message = llm_with_tools.invoke(messages)
        return {'messages': [message]}
    
    def take_action(state: AgentState) -> AgentState:
        """Execute tool calls from the LLM's response."""
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            if not t['name'] in tools_dict:
                result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
            else:
                result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}
    
    def evaluate_response(state: AgentState) -> AgentState:
        """Evaluate the final response using the evaluator LLM."""
        # Get the final AI response (last non-tool message)
        final_response = state['messages'][-1].content
        user_query = state['user_query']
        
        # Format the evaluation prompt
        eval_prompt = evaluation_prompt.format(
            user_query=user_query,
            ai_response=final_response
        )
        
        # Get evaluation from the evaluator LLM
        evaluation_message = _llm_evaluator.invoke([SystemMessage(content=eval_prompt)])
        response_text = evaluation_message.content

        # Extract score using markers
        try:
            score_match = re.search(r'<<<SCORE>>>(\d+\.?\d*)<<<END>>>', response_text)
            if score_match:
                evaluation_score = float(score_match.group(1))
            else:
                # Fallback parsing
                numbers = re.findall(r'\d+\.?\d*', response_text)
                evaluation_score = float(numbers[-1]) if numbers else 0.0
            
            evaluation_score = max(0.0, min(1.0, evaluation_score))
            
        except Exception as e:
            print(f"⚠️ Could not parse score: {e}")
            evaluation_score = 0.0

        
        return {
            'messages': state['messages'],  
            'evaluation_score': evaluation_score
        }

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_node("evaluator", evaluate_response)

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {True: "retriever_agent", False: "evaluator"}
    )
    graph.add_edge("retriever_agent", "llm")

    graph.add_edge("evaluator", END)

    graph.set_entry_point("llm")
    
    rag_agent = graph.compile()
    
    # Save the graph (only once)
    try:
        graph_image = rag_agent.get_graph().draw_mermaid_png()
        with open("rag_agent_graph_v2.png", "wb") as f:
            f.write(graph_image)
        print("✅ RAG agent graph saved as 'rag_agent_graph_v2.png'")
    except Exception as e:
        print(f"⚠️ Could not save RAG agent graph: {e}")
    
    print("✅ RAG agent created successfully!")
    return rag_agent


@sl.cache_resource
def create_rag_agent_v3(_llm, _retriever, _llm_evaluator):
    """
    Create RAG agent with evaluation step and conversation memory.
    
    Args:
        _llm: Language model instance for main responses
        _retriever: Document retriever instance
        _llm_evaluator: Language model instance for evaluation
        
    Returns:
        Compiled RAG agent graph with evaluation and memory
    """
    print("🤖 Creating RAG agent v3 with conversation memory...")
    
    @tool
    def retriever_tool(query: str) -> str:
        """This tool searches and returns the information from a pdf document."""
        docs = _retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the pdf document."
        
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Document {i+1}:\n{doc.page_content}")
        
        return "\n\n".join(results)
    
    tools = [retriever_tool]
    llm_with_tools = _llm.bind_tools(tools)
    
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        user_query: str  # Store original user query for evaluation
        evaluation_score: float  # Store evaluation score
        conversation_history: List[Dict[str, str]]  # Store previous conversations
    
    def should_continue(state: AgentState):
        """Check if the last message contains tool calls."""
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

    system_prompt = """
    You are an intelligent AI assistant who answers questions about the pdf document loaded into your knowledge base.
    Use the retriever tool available to answer questions. You can make multiple calls if needed.
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    Please always cite the specific parts of the documents you use in your answers (e.g. page, section).

    CONVERSATION CONTEXT GUIDELINES:
    - If previous conversation context is provided, use it to give more relevant and connected responses
    - You can reference previous questions or answers when it adds value to the current response
    - Always prioritize answering the current question thoroughly
    - If the current question relates to or builds upon previous conversations, acknowledge this connection
    - Maintain consistency with previous responses while providing new information for the current query
    """
    
    evaluation_prompt = """

    You are a top 1 percent expert evaluator tasked with assesing top notch scientists. 
    Be critical and demanding in your evaluation     
    You are a top 1 percent expert evaluator tasked with assesing top notch scientists.
     Be critical and demanding in your evaluation     
     as your reputation as a top expert is on the line and you might dissapear if this task is not done at your best capability.
    EVALUATION PROCESS:      
    
    1. EXTRACT MAIN QUESTIONS:     
    
    First, identify the core questions being asked in the user query (ignore supporting context).      
    
    2. EVALUATE EACH CRITERION (a score in the range of 0.00 to 0.20 each):      
    
    - Relevance: How well does the response address the user's question?           
    
    - Accuracy: Is the information provided correct and well-supported?          
    
    - Completeness: Does the response fully answer the question? Does it include examples?          
    - Clarity: Is the response clear and well-structured so as to be understood by a any professional     
     and it has zero ambiguity?          
     
     - Citation: Are there citations? are there of a good enough quality? are they relevant to the user's question?        
     
    3. APPLY STRICT STANDARDS:     
    - Be highly critical - most responses should score 0.20-0.60 after summing the 5 components.     
    - Only exceptional responses deserve 0.60+     
    - 0.85+ answers are 1 in a 100     
    - Common issues that reduce scores: vague language, missing examples, weak citations, incomplete answers      
    
    User Query: {user_query}     
    Answer: {ai_response}      
    
    THINK STEP BY STEP:      
    
    Step 1 - Main Questions: [Identify the core questions]      
    
    Step 2 - Relevance Score: [Evaluate and assign a score in the range 0.00 to 0.20]      
    
    Step 3 - Accuracy Score: [Evaluate and assign a score in the range 0.00 to 0.20]      
    
    Step 4 - Completeness Score: [Evaluate and assign a score in the range 0.00 to 0.20]      
    
    Step 5 - Clarity Score: [Evaluate and assign a score in the range 0.00 to 0.20]      
    
    Step 6 - Citations Score: [Evaluate and assign a score in the range 0.00 to 0.20]      
    
    Step 7 - Total Score: [Sum all scores = X.XX]

    Write the scores for each component with almost no text. Then write the sum of all (X.XX, e.g. 0.48).      

    IMPORTANT: After your analysis, you MUST end your response with exactly this format, indicating the sum of all criteria:
    <<<SCORE>>>X.XX<<<END>>>

    """
    
    tools_dict = {tool.name: tool for tool in tools}
    
    def call_llm(state: AgentState) -> AgentState:
        """Function to call the LLM with the current state and conversation history."""
        messages = list(state['messages'])
        
        # Build context-aware system prompt
        context_prompt = system_prompt
        
        # Add conversation history if available (simple approach)
        if 'conversation_history' in state and state['conversation_history']:
            history_context = "\n\nPrevious conversation context (for reference):\n"
            # Just add the last 2 conversations for context
            recent_conversations = state['conversation_history'][-2:]
            for i, conv in enumerate(recent_conversations, 1):
                history_context += f"Previous Q{i}: {conv['user_query']}\n"
                history_context += f"Previous A{i}: {conv['assistant_response'][:300]}...\n\n"
            
            context_prompt = f"{system_prompt}{history_context}"
        
        messages = [SystemMessage(content=context_prompt)] + messages
        message = llm_with_tools.invoke(messages)
        return {'messages': [message]}
    
    def take_action(state: AgentState) -> AgentState:
        """Execute tool calls from the LLM's response."""
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            if not t['name'] in tools_dict:
                result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
            else:
                result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}
    
    def evaluate_response(state: AgentState) -> AgentState:
        """Evaluate the final response using the evaluator LLM."""
        # Get the final AI response (last non-tool message)
        final_response = state['messages'][-1].content
        user_query = state['user_query']
        
        # Format the evaluation prompt
        eval_prompt = evaluation_prompt.format(
            user_query=user_query,
            ai_response=final_response
        )
        
        # Get evaluation from the evaluator LLM
        evaluation_message = _llm_evaluator.invoke([SystemMessage(content=eval_prompt)])
        response_text = evaluation_message.content

        # Extract score using markers
        try:
            score_match = re.search(r'<<<SCORE>>>(\d+\.?\d*)<<<END>>>', response_text)
            if score_match:
                evaluation_score = float(score_match.group(1))
            else:
                # Fallback parsing
                numbers = re.findall(r'\d+\.?\d*', response_text)
                evaluation_score = float(numbers[-1]) if numbers else 0.0
            
            evaluation_score = max(0.0, min(1.0, evaluation_score))
            
        except Exception as e:
            print(f"⚠️ Could not parse score: {e}")
            evaluation_score = 0.0

        
        return {
            'messages': state['messages'],  
            'evaluation_score': evaluation_score
        }

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_node("evaluator", evaluate_response)

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {True: "retriever_agent", False: "evaluator"}
    )
    graph.add_edge("retriever_agent", "llm")
    graph.add_edge("evaluator", END)
    graph.set_entry_point("llm")
    
    rag_agent = graph.compile()
    
    # Save the graph (only once)
    try:
        graph_image = rag_agent.get_graph().draw_mermaid_png()
        with open("rag_agent_graph_v3.png", "wb") as f:
            f.write(graph_image)
        print("✅ RAG agent v3 graph saved as 'rag_agent_graph_v3.png'")
    except Exception as e:
        print(f"⚠️ Could not save RAG agent graph: {e}")
    
    print("✅ RAG agent v3 created successfully!")
    return rag_agent

@sl.cache_resource
def initialize_llm_and_embeddings():
    """
    Initialize LLM and embeddings once per session.
    
    Args:
        use_finetuned (bool): Whether to use fine-tuned model if available
    
    Returns:
        tuple: (llm, embeddings)
    """
    
    # Your fine-tuned model ID
    #finetuned_model_id = "ft:gpt-4o-2024-08-06:personal::Bew7X3c0"
    finetuned_model_id = None

    # Default model fallback
    default_model = "gpt-4o"
    
    # Determine which model to use
    if  finetuned_model_id:
        model_to_use = finetuned_model_id
        model_type = "Fine-tuned"
    else:
        model_to_use = default_model
        model_type = "Base"
    
    try:
        # Initialize LLM with selected model
        llm = ChatOpenAI(
            model=model_to_use, 
            temperature=0,
            # Optional: Add other parameters
            max_tokens=None,  # Let the model decide
            timeout=30,       # 30 second timeout
        )
        
        # Test the model with a simple query to ensure it works
        test_response = llm.invoke("Hello")
        
        print(f"✅ {model_type} model loaded successfully: {model_to_use}")
        
    except Exception as e:
        print(f"⚠️ Failed to load {model_type.lower()} model: {e}")
        
        # Fallback to default model if fine-tuned fails
        if  model_to_use != default_model:
            print(f"🔄 Falling back to base model: {default_model}")
            llm = ChatOpenAI(model=default_model, temperature=0)
            model_to_use = default_model
            model_type = "Base (Fallback)"
        else:
            raise e
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")    
    print("✅ LLM and embeddings initialized!")

    return llm, embeddings


@sl.cache_resource
def initialize_llm_and_embeddings_v2():
    """
    Initialize LLM, evaluator LLM, and embeddings once per session.
    
    Returns:
        tuple: (llm, llm_evaluator, embeddings)
    """
    
    # Your fine-tuned model ID
    #finetuned_model_id = "ft:gpt-4o-2024-08-06:personal::Bew7X3c0"
    finetuned_model_id = None
    default_model = "gpt-4o"
    
    # Determine which model to use for main LLM
    if finetuned_model_id:
        model_to_use = finetuned_model_id
        model_type = "Fine-tuned"
    else:
        model_to_use = default_model
        model_type = "Base"
    
    try:
        # Initialize main LLM
        llm = ChatOpenAI(
            model=model_to_use, 
            temperature=0,
            max_tokens=None,
            timeout=30,
        )
        
        # Test the model
        test_response = llm.invoke("Hello")
        print(f"✅ {model_type} model loaded successfully: {model_to_use}")
        
    except Exception as e:
        print(f"⚠️ Failed to load {model_type.lower()} model: {e}")
        
        # Fallback to default model if fine-tuned fails
        if model_to_use != default_model:
            print(f"🔄 Falling back to base model: {default_model}")
            llm = ChatOpenAI(model=default_model, temperature=0)
        else:
            raise e
    
    # Initialize evaluator LLM (GPT-3.5 Turbo), max tokes set  to minimize the cost
    # we only need a simple result like 0.35 
    # this model could be changed to llama or some other open llm 
    # as this task is relatively simple and finetuning could be used for optimization

    llm_evaluator = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=200,
        timeout=10,
        frequency_penalty=2.0,  # ✅ Reduce repetitive patterns
        presence_penalty=2.0
    )

    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("✅ All models and embeddings initialized successfully!")
    return llm, llm_evaluator, embeddings

@sl.cache_data
def load_and_process_pdf(pdf_path: str):
    """
    Load and process PDF once per session.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of processed document chunks
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
    """
    print(f"📄 Loading and processing PDF: {pdf_path}")
    
    # Safety measure
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    pdf_loader = PyPDFLoader(pdf_path)
    
    try:
        pages = pdf_loader.load()
        print(f"✅ PDF loaded successfully with {len(pages)} pages")
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        raise
    
    # Chunking Process
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    pages_split = text_splitter.split_documents(pages)
    print(f"✅ PDF processed into {len(pages_split)} chunks")
    return pages_split


@sl.cache_resource
def create_vector_store(_pages_split, _embeddings):
    """
    Create vector store once per session.
    
    Args:
        _pages_split: Processed document chunks
        _embeddings: Embeddings model instance
        
    Returns:
        ChromaDB vector store instance
        
    Raises:
        Exception: If vector store creation fails
    """
    print("🗂️ Creating vector store...")
    
    persist_directory = os.getcwd()
    collection_name = "pdf_document"
    
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    try:
        vectorstore = Chroma.from_documents(
            documents=_pages_split,
            embedding=_embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print("✅ ChromaDB vector store created successfully!")
        return vectorstore
    except Exception as e:
        print(f"❌ Error setting up ChromaDB: {str(e)}")
        raise
