####################################  IMPORTS #######################################

from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
import streamlit as sl
from codecarbon import EmissionsTracker
import tiktoken
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from flask import Flask, Response
import threading
import time
from typing import List
import re
import json
from datetime import datetime
from IPython.display import display, Image
from helper_functions import Monitoring, Logging, Metrics, LatencyTracker, init_metrics
from Langgraph_Agent import initialize_llm_and_embeddings, create_vector_store, load_and_process_pdf, create_rag_agent
from Langgraph_Agent import initialize_llm_and_embeddings_v2, create_rag_agent_v2, create_rag_agent_v3
import argparse

########################  FUNCTIONS  #######################################

@sl.cache_resource
def initialize_monitoring():
    """Initialize monitoring components once per session."""
    monitor = Monitoring()
    
    # Create session-specific Gauge metrics
    from prometheus_client import Gauge
    try:
        CURRENT_SESSION_QUERIES = Gauge('pdf_rag_session_queries_current', 'Current session query count')
        CURRENT_SESSION_AVG_SATISFACTION = Gauge('pdf_rag_session_avg_satisfaction', 'Current session average satisfaction')
        CURRENT_SESSION_AVG_ADHERENCE = Gauge('pdf_rag_session_avg_adherence', 'Current session average context adherence')
        print("✅ Session metrics created successfully!")
    except ValueError as e:
        print(f"⚠️ Session metrics already exist: {e}")
        # Get existing metrics from registry
        CURRENT_SESSION_QUERIES = None
        CURRENT_SESSION_AVG_SATISFACTION = None
        CURRENT_SESSION_AVG_ADHERENCE = None
        
        for collector in REGISTRY._collector_to_names.keys():
            if hasattr(collector, '_name'):
                if collector._name == 'pdf_rag_session_queries_current':
                    CURRENT_SESSION_QUERIES = collector
                elif collector._name == 'pdf_rag_session_avg_satisfaction':
                    CURRENT_SESSION_AVG_SATISFACTION = collector
                elif collector._name == 'pdf_rag_session_avg_adherence':
                    CURRENT_SESSION_AVG_ADHERENCE = collector
    
    # Start metrics server in background thread (only once)
    metrics_thread = threading.Thread(target=monitor.run_metrics_server, daemon=True)
    metrics_thread.start()
    
    return monitor, CURRENT_SESSION_QUERIES, CURRENT_SESSION_AVG_SATISFACTION, CURRENT_SESSION_AVG_ADHERENCE

########################  LOAD COMPONENTS  #######################################

load_dotenv()

# Clear metrics only once per session, then reinitialize
if 'metrics_cleared' not in sl.session_state:
    print("🧹 Clearing existing metrics...")
    Monitoring.clear_prometheus_metrics()
    sl.session_state.metrics_cleared = True
    
# Initialize metrics after clearing
print("🔧 Initializing metrics...")
pdf_rag_metrics = init_metrics()
print("✅ PDF RAG metrics initialized!")

# Initialize components
monitor, CURRENT_SESSION_QUERIES, CURRENT_SESSION_AVG_SATISFACTION, CURRENT_SESSION_AVG_ADHERENCE = initialize_monitoring()

# Import functions - direct references to static methods
log_metrics_to_prometheus = Monitoring.log_metrics_to_prometheus
log_session_metrics_to_prometheus = Monitoring.log_session_metrics_to_prometheus
save_training_data = Logging.save_training_data
calculate_context_adherence = Metrics.calculate_context_adherence
calculate_semantic_overlap = Metrics.calculate_semantic_overlap
count_tokens = Metrics.count_tokens
calculate_cost = Metrics.calculate_cost
enhanced_retriever_tool = Metrics.enhanced_retriever_tool

#estimations to normalize satisfaction per dollar and quality per dollar to be in the range 0.00 to 1.00
estimated_min_cost, _, _= calculate_cost( 30 , 150 ) 
max_satisfaction_per_dollar = 5.0 / estimated_min_cost
max_quality_per_dollar = 1.0 / estimated_min_cost
price_per_kwh = 0.192 #average canada, reference: https://www.energyhub.org/electricity-prices/

# Test metrics on startup
print("🧪 Testing metrics initialization...")
try:
    # Run a simple test
    from helper_functions import test_metrics
    if test_metrics():
        print("✅ Metrics system is working correctly!")
    else:
        print("⚠️ Metrics test failed - check configuration")
except Exception as e:
    print(f"⚠️ Could not test metrics: {e}")
    import traceback
    traceback.print_exc()


# Call debug function
Metrics.debug_metrics()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture_version", "-v", type=int, choices=[1,2,3])

    args = parser.parse_args()
    version = args.architecture_version

    # Initialize all components (cached - only runs once)
    if version == 1:
        llm, embeddings = initialize_llm_and_embeddings()
    else:
        llm, llm_evaluator, embeddings = initialize_llm_and_embeddings_v2()

    # Load PDF and create components (cached - only runs once)
    pdf_path = "random_machine_learing.pdf"
    pages_split = load_and_process_pdf(pdf_path)
    vectorstore = create_vector_store(pages_split, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    if version == 1:
        rag_agent = create_rag_agent(llm, retriever)
    elif version == 2:
        rag_agent = create_rag_agent_v2(llm, retriever, llm_evaluator)
    else:  # version == 3
        rag_agent = create_rag_agent_v3(llm, retriever, llm_evaluator)

    sl.header("welcome to the 📝PDF bot")
    version_descriptions = {
        1: "🤖 Basic RAG Agent",
        2: "🤖 RAG Agent with LLM Evaluator", 
        3: "🤖 RAG Agent with LLM Evaluator + Conversation Memory"
    }
    sl.write(f"{version_descriptions.get(version, '🤖')} You can chat by entering your queries")
    
    # Initialize session state for tracking
    if 'total_energy' not in sl.session_state:
        sl.session_state.total_energy = 0
        sl.session_state.total_queries = 0
        sl.session_state.total_cost = 0
        sl.session_state.total_llm_evaluator_score = 0
        sl.session_state.total_tokens = 0
        sl.session_state.total_adherence = 0
        sl.session_state.total_latency = 0
        sl.session_state.total_carbon = 0
        sl.session_state.total_satisfaction = 0
        sl.session_state.satisfaction_scores = []
        sl.session_state.satisfaction_score = 0
        sl.session_state.awaiting_score = False
        sl.session_state.current_metrics = None
        sl.session_state.score_submitted = False
    
    # Initialize conversation memory for version 3 only
    if version == 3 and 'conversation_memory' not in sl.session_state:
        sl.session_state.conversation_memory = []  # Store last 3 conversations
    
    query = sl.text_input('Enter some text')
    
    # Display conversation memory for version 3
    if version == 3 and sl.session_state.get('conversation_memory'):
        with sl.expander("💭 Conversation Memory (Last 3 exchanges)"):
            for i, conv in enumerate(reversed(sl.session_state.conversation_memory), 1):
                sl.write(f"**Exchange {len(sl.session_state.conversation_memory) - i + 1}:**")
                sl.write(f"*Q: {conv['user_query'][:100]}...*")
                sl.write(f"*A: {conv['assistant_response'][:200]}...*")
                sl.write("---")
    
    # Only process new queries when not awaiting score AND query is different from last processed
    if 'last_processed_query' not in sl.session_state:
        sl.session_state.last_processed_query = ""
    
    if (query and 
        not sl.session_state.awaiting_score and 
        query != sl.session_state.last_processed_query):
        
        # Initialize trackers
        tracker = EmissionsTracker(
            project_name="langgraph_pdf_chatbot",
            output_dir="./emissions_data",
            save_to_file=True,
            log_level="warning"
        )
        
        latency_tracker = LatencyTracker()
        
        # Start tracking
        tracker.start()
        latency_tracker.start()
        
        with sl.spinner("Processing your question..."):
            # Step 1: Start processing
            latency_tracker.log_step("start_processing")
            
            # Step 2: Retrieve documents
            retrieved_docs_text, raw_retrieved_docs = enhanced_retriever_tool(query, retriever)
            latency_tracker.log_step("retrieval_complete")
            
            # Step 3: LLM inference
            if version == 1:
                initial_state = {"messages": [HumanMessage(content=query)]}                
            elif version == 2:
                initial_state = {
                    'messages': [HumanMessage(content=query)], 
                    'user_query': query,
                    'evaluation_score': 0.0
                }
            else:  # version == 3
                initial_state = {
                    'messages': [HumanMessage(content=query)], 
                    'user_query': query,
                    'evaluation_score': 0.0,
                    'conversation_history': sl.session_state.conversation_memory.copy()
                }

            result = rag_agent.invoke(initial_state)
            response = result['messages'][-1].content
            evaluation_score = 0.00 if version == 1 else result['evaluation_score']    
            latency_tracker.log_step("llm_complete")
            
            # Store conversation in memory (version 3 only)
            if version == 3:
                current_conversation = {
                    'user_query': query,
                    'assistant_response': response
                }
                sl.session_state.conversation_memory.append(current_conversation)
                
                # Keep only last 3 conversations
                if len(sl.session_state.conversation_memory) > 3:
                    sl.session_state.conversation_memory = sl.session_state.conversation_memory[-3:]
        
        # Stop tracking and get results
        emissions_data = tracker.stop()
        latency_tracker.log_step("end_processing")
        
        # Calculate metrics
        input_tokens = count_tokens(query)
        output_tokens = count_tokens(response)
        total_cost, input_cost, output_cost = calculate_cost(input_tokens, output_tokens)
        total_cost = total_cost + ( emissions_data*price_per_kwh )
        
        # Context adherence metrics
        word_adherence = calculate_context_adherence(response, raw_retrieved_docs)
        semantic_adherence = calculate_semantic_overlap(response, raw_retrieved_docs)
        combined_adherence = (word_adherence + semantic_adherence) / 2
        
        # Latency metrics
        retrieval_time = latency_tracker.get_duration("start_processing", "retrieval_complete")
        llm_time = latency_tracker.get_duration("retrieval_complete", "llm_complete")
        total_latency = latency_tracker.get_total_time()
        
        # Store current metrics for after satisfaction score
        sl.session_state.current_metrics = {
            'query': query,
            'response': response,
            'emissions_data': emissions_data,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_cost': total_cost,
            'combined_adherence': combined_adherence,
            'total_latency': total_latency,
            'retrieval_time': retrieval_time,
            'llm_time': llm_time,
            'llm_evaluator_score': evaluation_score
        }
        
        # Log to Prometheus
        log_metrics_to_prometheus(sl.session_state.current_metrics)
        
        sl.write("**Answer:**")
        sl.write(response)
        
        # Store the processed query to prevent reprocessing
        sl.session_state.last_processed_query = query
        
        # Set flag to await satisfaction score
        sl.session_state.awaiting_score = True
        sl.session_state.score_submitted = False

    # User satisfaction scoring section
    if sl.session_state.awaiting_score:
        sl.write("---")
        
        # Only show the rating form if score hasn't been submitted yet
        if not sl.session_state.score_submitted:
            sl.subheader("📊 Rate this answer")
            sl.write("Please rate the quality of this answer from 0 (very poor) to 5 (excellent):")
            
            # Create a unique key for each new question to reset selectbox
            if 'selectbox_key' not in sl.session_state:
                sl.session_state.selectbox_key = 0
            
            satisfaction_score = sl.selectbox(
                "Your satisfaction score:",
                options=[None, 0, 1, 2, 3, 4, 5],
                format_func=lambda x: "Select a score..." if x is None else f"{x} - {'⭐' * int(x) if x > 0 else '❌'}",
                key=f"satisfaction_{sl.session_state.selectbox_key}"
            )
            
            # Submit button to confirm the rating
            if sl.button("Submit Rating"):
                if satisfaction_score is not None:
                    # Mark score as submitted to prevent reprocessing
                    sl.session_state.score_submitted = True
                    
                    sl.session_state.satisfaction_score = satisfaction_score
                    log_metrics_to_prometheus(sl.session_state.current_metrics, satisfaction_score)

                    save_training_data(
                        query=sl.session_state.current_metrics['query'],
                        response=sl.session_state.current_metrics['response'],
                        satisfaction_score=satisfaction_score,
                        context_adherence=sl.session_state.current_metrics['combined_adherence']
                    )          

                    # Update session totals with stored metrics
                    metrics = sl.session_state.current_metrics
                    
                    sl.session_state.total_energy += metrics['emissions_data']
                    sl.session_state.total_queries += 1
                    sl.session_state.total_cost += metrics['total_cost']
                    sl.session_state.total_llm_evaluator_score += metrics['llm_evaluator_score']
                    sl.session_state.total_tokens += (metrics['input_tokens'] + metrics['output_tokens'])
                    sl.session_state.total_adherence += metrics['combined_adherence']
                    sl.session_state.total_latency += metrics['total_latency']
                    sl.session_state.total_carbon += metrics['emissions_data']
                    sl.session_state.total_satisfaction += satisfaction_score
                    sl.session_state.satisfaction_scores.append(satisfaction_score)
                    
                    CURRENT_SESSION_QUERIES.set(sl.session_state.total_queries)
                    CURRENT_SESSION_AVG_SATISFACTION.set(sl.session_state.total_satisfaction / sl.session_state.total_queries)
                    CURRENT_SESSION_AVG_ADHERENCE.set(sl.session_state.total_adherence / sl.session_state.total_queries)
                    
                    # Increment selectbox key for next question
                    sl.session_state.selectbox_key += 1
                    
                    # Force a rerun to update the display
                    sl.rerun()
                else:
                    sl.error("Please select a rating before submitting.")
        
        else:
            # Show only the confirmation message when score is submitted
            last_score = sl.session_state.satisfaction_scores[-1] if sl.session_state.satisfaction_scores else 0
            sl.success(f"✅ Rating submitted: {last_score}/5 {'⭐' * int(last_score) if last_score > 0 else '❌'}")

    # Display metrics only after score is submitted
    if sl.session_state.awaiting_score and sl.session_state.score_submitted:
        # Button to continue to next question
        if sl.button("Ask Another Question", type="primary"):
            # Reset for next query
            sl.session_state.awaiting_score = False
            sl.session_state.current_metrics = None
            sl.session_state.last_processed_query = ""
            sl.session_state.score_submitted = False
            sl.rerun()
        
        # Calculate averages
        avg_energy = sl.session_state.total_energy / sl.session_state.total_queries
        avg_cost = sl.session_state.total_cost / sl.session_state.total_queries
        avg_llm_evaluator_score= sl.session_state.total_llm_evaluator_score / sl.session_state.total_queries
        avg_tokens = sl.session_state.total_tokens / sl.session_state.total_queries
        avg_adherence = sl.session_state.total_adherence / sl.session_state.total_queries
        avg_latency = sl.session_state.total_latency / sl.session_state.total_queries
        avg_carbon = sl.session_state.total_carbon / sl.session_state.total_queries
        avg_satisfaction = sl.session_state.total_satisfaction / sl.session_state.total_queries

        # Calculate performance metrics
        efficiency_score = avg_satisfaction / avg_cost if avg_cost > 0 else 0
        quality_efficiency = avg_adherence / avg_cost if avg_cost > 0 else 0
        
        # Calculate satisfaction distribution
        total_scores = len(sl.session_state.satisfaction_scores)
        if total_scores > 0:
            excellent_count = sl.session_state.satisfaction_scores.count(5) + sl.session_state.satisfaction_scores.count(4)
            good_count = sl.session_state.satisfaction_scores.count(3)
            poor_count = sl.session_state.satisfaction_scores.count(0) + sl.session_state.satisfaction_scores.count(1) + sl.session_state.satisfaction_scores.count(2)
            
            excellent_pct = (excellent_count / total_scores) * 100
            good_pct = (good_count / total_scores) * 100
            poor_pct = (poor_count / total_scores) * 100
        else:
            excellent_pct = good_pct = poor_pct = 0

        # Prepare session data for Prometheus
        session_data = {
            'tokens_query': sl.session_state.current_metrics['input_tokens'] + sl.session_state.current_metrics['output_tokens'],
            'carbon_footprint': sl.session_state.current_metrics['emissions_data']*1000000,
            'energy_per_query': sl.session_state.current_metrics['emissions_data'],
            'latency_query': sl.session_state.current_metrics['total_latency'],

            'context_adherence_query': sl.session_state.current_metrics['combined_adherence'], 
            'user_satisfaction': sl.session_state.satisfaction_score,
            'cost_per_query': sl.session_state.current_metrics['total_cost'],

            'satisfaction_per_dollar': (sl.session_state.satisfaction_scores[-1]/ sl.session_state.current_metrics['total_cost'])/ max_satisfaction_per_dollar,
            'quality_per_dollar': (sl.session_state.current_metrics['combined_adherence']/ sl.session_state.current_metrics['total_cost'])/ max_quality_per_dollar,
            'llm_score': sl.session_state.current_metrics['llm_evaluator_score'],
            
        }
        
        # Send session metrics to Prometheus
        log_session_metrics_to_prometheus(session_data)

        # Display metrics sections (same as before, but now also sending to Prometheus)
        sl.write("---")
        
        # Session averages
        sl.subheader("Session Averages")
        col1, col2, col3 = sl.columns(3)
        with col1:
            sl.metric("Total Tokens", sl.session_state.total_tokens)
        with col2:
            sl.metric("Total Queries", sl.session_state.total_queries)
        with col3:
            sl.metric("Avg Tokens per Query", f"{avg_tokens:.1f}")
        
        # Session performance
        sl.subheader("Session Performance")
        col1, col2, col3, col4 = sl.columns(4)
        with col1:
            sl.metric("Avg Context Adherence", f"{avg_adherence:.3f}")
        with col2:
            sl.metric("Avg Latency", f"{avg_latency:.3f}s")
        with col3:
            sl.metric("Avg Carbon per Query", f"{avg_carbon * 1000000:.3f} mg CO2eq")
        with col4:
            sl.metric("Avg User Satisfaction", f"{avg_satisfaction:.2f}/5", f"{'⭐' * int(avg_satisfaction) if avg_satisfaction > 0 else '❌'}")

        # Session costs
        sl.subheader("Session Costs")
        col1, col2 = sl.columns(2)
        with col1:
            sl.metric("Avg Cost per Query", f"${avg_cost:.6f}")
        with col2:
            sl.metric("Avg Energy per Query", f"{avg_energy:.8f} kWh")

        # Performance vs Cost Analysis
        sl.subheader("Performance  Analysis")
        col1, col2, col3 = sl.columns(3)
        with col1:
            sl.metric("Satisfaction per Dollar", f"{efficiency_score/max_satisfaction_per_dollar:.2f}", help="User satisfaction score divided by cost")
        with col2:
            sl.metric("Quality per Dollar", f"{quality_efficiency/max_quality_per_dollar:.2f}", help="Context adherence divided by cost")
        with col3:
            sl.metric("LLM evaluator score (0.00 - 1.00)", f"{avg_llm_evaluator_score:.2f}")


        # Satisfaction Distribution
        if len(sl.session_state.satisfaction_scores) > 1:
            sl.subheader("Satisfaction Distribution")
            col1, col2, col3 = sl.columns(3)
            with col1:
                sl.metric("Excellent Answers", f"{excellent_pct:.1f}%", "Scores 4-5")
            with col2:
                sl.metric("Good Answers", f"{good_pct:.1f}%", "Score 3")
            with col3:
                sl.metric("Poor Answers", f"{poor_pct:.1f}%", "Scores 0-2")

    # Show instruction when awaiting score (only if not submitted)
    elif sl.session_state.awaiting_score and not sl.session_state.score_submitted:
        sl.info("👆 Please rate the answer above before viewing the metrics.")

main()