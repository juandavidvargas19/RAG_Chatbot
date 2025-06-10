#import Essential dependencies
#use the following to run: streamlit run streamlit_app.py

import streamlit as sl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

#doesn't remember conversation, add memory

from dotenv import load_dotenv
load_dotenv()


###############

import time
from typing import List
import re
from codecarbon import EmissionsTracker
import tiktoken


def calculate_context_adherence(response: str, retrieved_docs: List[str]) -> float:
    """
    Calculate context adherence score based on overlap between response and retrieved documents.
    Returns a score between 0 and 1.
    """
    if not response or not retrieved_docs:
        return 0.0
    
    # Combine all retrieved documents
    combined_context = " ".join(retrieved_docs)
    
    # Convert to lowercase for comparison
    response_lower = response.lower()
    context_lower = combined_context.lower()
    
    # Split into words and remove common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    
    # Extract meaningful words from response
    response_words = set(word.strip('.,!?;:"()[]') for word in response_lower.split() if len(word) > 3 and word not in stopwords)
    
    # Count how many response words are found in context
    matching_words = 0
    for word in response_words:
        if word in context_lower:
            matching_words += 1
    
    # Calculate adherence score
    if len(response_words) == 0:
        return 0.0
    
    adherence_score = matching_words / len(response_words)
    return min(adherence_score, 1.0)  # Cap at 1.0

def calculate_semantic_overlap(response: str, retrieved_docs: List[str]) -> float:
    """
    Calculate semantic overlap using common phrases and n-grams.
    """
    if not response or not retrieved_docs:
        return 0.0
    
    combined_context = " ".join(retrieved_docs).lower()
    response_lower = response.lower()
    
    # Look for 2-gram and 3-gram overlaps
    response_ngrams = set()
    words = response_lower.split()
    
    # Generate 2-grams and 3-grams
    for i in range(len(words) - 1):
        response_ngrams.add(f"{words[i]} {words[i+1]}")
    for i in range(len(words) - 2):
        response_ngrams.add(f"{words[i]} {words[i+1]} {words[i+2]}")
    
    # Count matches
    matches = sum(1 for ngram in response_ngrams if ngram in combined_context)
    
    if len(response_ngrams) == 0:
        return 0.0
    
    return min(matches / len(response_ngrams), 1.0)

class LatencyTracker:
    def __init__(self):
        self.start_time = None
        self.timings = {}
    
    def start(self):
        self.start_time = time.perf_counter()
        self.timings = {}
    
    def log_step(self, step_name: str):
        current_time = time.perf_counter()
        if self.start_time is None:
            self.start_time = current_time
        
        self.timings[step_name] = current_time - self.start_time
    
    def get_duration(self, start_step: str, end_step: str) -> float:
        if start_step in self.timings and end_step in self.timings:
            return self.timings[end_step] - self.timings[start_step]
        return 0.0
    
    def get_total_time(self) -> float:
        if not self.timings:
            return 0.0
        return max(self.timings.values())

# Modified retriever_tool to capture retrieved documents
def enhanced_retriever_tool(query: str, retriever) -> tuple:
    """
    Enhanced retriever that returns both the formatted result and raw documents.
    """
    docs = retriever.invoke(query)
    
    if not docs:
        return "I found no relevant information in the document.", []
    
    results = []
    raw_docs = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
        raw_docs.append(doc.page_content)
    
    return "\n\n".join(results), raw_docs




#################################

def count_tokens(text, model="gpt-4o"):
    """Count tokens in text for the specified model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base encoding for newer models
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def calculate_cost(input_tokens, output_tokens, model="gpt-4o"):
    """Calculate cost based on OpenAI pricing for GPT-4o."""
    # GPT-4o pricing (as of 2024)
    input_cost_per_1k = 0.005   # $0.005 per 1K input tokens
    output_cost_per_1k = 0.015  # $0.015 per 1K output tokens
    
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    
    return input_cost + output_cost, input_cost, output_cost




################


#function to load the vectordatabase
def load_knowledgeBase():
        embeddings=OpenAIEmbeddings()
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)  #do this change
        return db
        
#function to load the OPENAI LLM
def load_llm():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        return llm

#creating prompt template using langchain
def load_prompt():
        prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt


def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("./random machine learing pdf.pdf")
docs=loader.load()

if __name__ == '__main__':
    sl.header("welcome to the 📝PDF bot")
    sl.write("🤖 You can chat by Entering your queries ")
    
    # Initialize session state for tracking
    if 'total_energy' not in sl.session_state:
        sl.session_state.total_energy = 0
        sl.session_state.total_queries = 0
        sl.session_state.total_cost = 0
        sl.session_state.total_tokens = 0
        sl.session_state.total_adherence = 0
        sl.session_state.total_latency = 0
        sl.session_state.total_carbon = 0
        sl.session_state.total_satisfaction = 0
        sl.session_state.satisfaction_scores = []
        sl.session_state.awaiting_score = False
        sl.session_state.current_metrics = None
        sl.session_state.score_submitted = False
    
    # Load your existing components
    knowledgeBase = load_knowledgeBase()
    llm = load_llm()
    prompt = load_prompt()
    
    query = sl.text_input('Enter some text')
    
    # Only process new queries when not awaiting score AND query is different from last processed
    if 'last_processed_query' not in sl.session_state:
        sl.session_state.last_processed_query = ""
    
    if (query and 
        not sl.session_state.awaiting_score and 
        query != sl.session_state.last_processed_query):
        
        # Initialize trackers
        tracker = EmissionsTracker(
            project_name="pdf_chatbot",
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
            
            # Step 2: Getting similar embeddings (your existing retrieval logic)
            similar_embeddings = knowledgeBase.similarity_search(query)
            raw_retrieved_docs = [doc.page_content for doc in similar_embeddings]

            similar_embeddings = FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings())
            
            # Use enhanced retriever for better metrics
            latency_tracker.log_step("retrieval_complete")
            
            # Step 3: Creating the chain and getting response (your existing logic)
            retriever = similar_embeddings.as_retriever()


            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            response = rag_chain.invoke(query)
            latency_tracker.log_step("llm_complete")
        
        # Stop tracking and get results
        emissions_data = tracker.stop()
        latency_tracker.log_step("end_processing")
        
        # Calculate carbon footprint (emissions_data is in kg CO2eq)
        carbon_footprint_g = emissions_data * 1000  # Convert kg to grams
        carbon_footprint_mg = emissions_data * 1000000  # Convert kg to milligrams
        
        # Calculate metrics
        input_tokens = count_tokens(query)
        output_tokens = count_tokens(response)
        total_cost, input_cost, output_cost = calculate_cost(input_tokens, output_tokens)
        
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
            'emissions_data': emissions_data,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_cost': total_cost,
            'combined_adherence': combined_adherence,
            'total_latency': total_latency,
            'carbon_footprint_mg': carbon_footprint_mg
        }
        
        sl.write("**Answer:**")
        sl.write(response)
        
        # Store the processed query to prevent reprocessing
        sl.session_state.last_processed_query = query
        
        # Set flag to await satisfaction score
        sl.session_state.awaiting_score = True
        sl.session_state.score_submitted = False  # Reset score submission flag

    # User satisfaction scoring section
    if sl.session_state.awaiting_score and not sl.session_state.score_submitted:
        sl.write("---")
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
                
                # Update session totals with stored metrics
                metrics = sl.session_state.current_metrics
                
                sl.session_state.total_energy += metrics['emissions_data']
                sl.session_state.total_queries += 1
                sl.session_state.total_cost += metrics['total_cost']
                sl.session_state.total_tokens += (metrics['input_tokens'] + metrics['output_tokens'])
                sl.session_state.total_adherence += metrics['combined_adherence']
                sl.session_state.total_latency += metrics['total_latency']
                sl.session_state.total_carbon += metrics['emissions_data']
                sl.session_state.total_satisfaction += satisfaction_score
                sl.session_state.satisfaction_scores.append(satisfaction_score)
                
                # Increment selectbox key for next question
                sl.session_state.selectbox_key += 1
                
                sl.success(f"Rating submitted: {satisfaction_score}/5")
            else:
                sl.error("Please select a rating before submitting.")
    
    # Display metrics only after score is submitted
    if sl.session_state.awaiting_score and sl.session_state.score_submitted:
        # Calculate averages
        avg_energy = sl.session_state.total_energy / sl.session_state.total_queries
        avg_cost = sl.session_state.total_cost / sl.session_state.total_queries
        avg_tokens = sl.session_state.total_tokens / sl.session_state.total_queries
        avg_adherence = sl.session_state.total_adherence / sl.session_state.total_queries
        avg_latency = sl.session_state.total_latency / sl.session_state.total_queries
        avg_carbon = sl.session_state.total_carbon / sl.session_state.total_queries
        avg_satisfaction = sl.session_state.total_satisfaction / sl.session_state.total_queries
        
        metrics = sl.session_state.current_metrics
        current_satisfaction = sl.session_state.satisfaction_scores[-1]  # Get the last submitted score
        
        # Display current query metrics
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
        sl.subheader("Performance vs Cost Analysis")
        col1, col2 = sl.columns(2)
        with col1:
            efficiency_score = avg_satisfaction / avg_cost if avg_cost > 0 else 0
            sl.metric("Satisfaction per Dollar", f"{efficiency_score:.0f}", help="User satisfaction score divided by cost")
        with col2:
            quality_efficiency = avg_adherence / avg_cost if avg_cost > 0 else 0
            sl.metric("Quality per Dollar", f"{quality_efficiency:.0f}", help="Context adherence divided by cost")
        
        # Satisfaction Distribution
        if len(sl.session_state.satisfaction_scores) > 1:
            sl.subheader("Satisfaction Distribution")
            col1, col2, col3 = sl.columns(3)
            with col1:
                excellent_count = sl.session_state.satisfaction_scores.count(5) + sl.session_state.satisfaction_scores.count(4)
                excellent_pct = (excellent_count / len(sl.session_state.satisfaction_scores)) * 100
                sl.metric("Excellent Answers", f"{excellent_pct:.1f}%", "Scores 4-5")
            with col2:
                good_count = sl.session_state.satisfaction_scores.count(3)
                good_pct = (good_count / len(sl.session_state.satisfaction_scores)) * 100
                sl.metric("Good Answers", f"{good_pct:.1f}%", "Score 3")
            with col3:
                poor_count = sl.session_state.satisfaction_scores.count(0) + sl.session_state.satisfaction_scores.count(1) + sl.session_state.satisfaction_scores.count(2)
                poor_pct = (poor_count / len(sl.session_state.satisfaction_scores)) * 100
                sl.metric("Poor Answers", f"{poor_pct:.1f}%", "Scores 0-2")
        
        # Button to continue to next question
        if sl.button("Ask Another Question"):
            # Reset for next query
            sl.session_state.awaiting_score = False
            sl.session_state.current_metrics = None
            sl.session_state.last_processed_query = ""  # Clear to allow new queries
            sl.session_state.score_submitted = False
            sl.rerun()
    
    # Show instruction when awaiting score
    elif sl.session_state.awaiting_score and not sl.session_state.score_submitted:
        sl.info("👆 Please rate the answer above before viewing the metrics.")
        