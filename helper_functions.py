"""
PDF RAG Application Helper Functions

This module provides monitoring, logging, metrics calculation, and latency tracking
functionality for a PDF-based Retrieval-Augmented Generation (RAG) system.

Classes:
    - Logging: Handles training data persistence
    - Monitoring: Manages Prometheus metrics and monitoring server
    - Metrics: Provides utilities for context adherence and token/cost calculations
    - LatencyTracker: Tracks execution timing across different processing steps
"""

import json
import socket
import threading
import time
from datetime import datetime
from typing import List, Tuple

import tiktoken
from flask import Flask, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, REGISTRY, Counter, Gauge, Histogram,
    generate_latest
)

# Global variable to store metrics - will be initialized by get_or_create_metrics()
_METRICS_INITIALIZED = False
_METRICS = {}

def get_or_create_metrics():
    """Get or create Prometheus metrics (singleton pattern)."""
    global _METRICS_INITIALIZED, _METRICS
    
    if not _METRICS_INITIALIZED:
        try:
            # Initialize all PDF RAG metrics
            _METRICS = {
                # Individual query metrics
                'QUERY_COUNTER': Counter('pdf_rag_queries_total', 'Total number of queries processed'),
                'QUERY_DURATION': Histogram('pdf_rag_query_duration_seconds', 'Time spent processing queries',
                                          buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]),
                'TOKEN_USAGE': Counter('pdf_rag_tokens_total', 'Total tokens used', ['token_type']),
                'COST_TOTAL': Counter('pdf_rag_cost_total', 'Total cost incurred in USD'),
                'CONTEXT_ADHERENCE': Histogram('pdf_rag_context_adherence', 'Context adherence scores',
                                             buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),

                'RETRIEVAL_TIME': Histogram('pdf_rag_retrieval_duration_seconds', 'Time spent on document retrieval'),
                'LLM_TIME': Histogram('pdf_rag_llm_duration_seconds', 'Time spent on LLM inference'),
                
                # Session-level metrics (Gauges)

                'TOKENS_PER_QUERY': Gauge('pdf_rag_tokens_per_query', 'tokens per query'),
                'CONTEXT_ADHERENCE_PER_QUERY': Gauge('pdf_rag_context_adherence_per_query', 'context adherence per query'),


                'CARBON_FOOTPRINT': Gauge('pdf_rag_carbon_mg_per_query', 'carbon emissions per query in session'),
                'USER_SATISFACTION': Gauge('pdf_rag_user_satisfaction', 'user satisfaction per response'),
                'COST_PER_QUERY': Gauge('pdf_rag_cost_per_query', 'cost per query in dollars'),
                'ENERGY_PER_QUERY': Gauge('pdf_rag_energy_kwh_per_query', 'energy per query in kWh'),
                'LATENCY_PER_QUERY': Gauge('pdf_rag_latency_per_query', 'latency query in seconds'),
                
                # Performance analysis metrics (Gauges)
                'SATISFACTION_PER_DOLLAR': Gauge('pdf_rag_satisfaction_per_dollar', 'User satisfaction per dollar spent'),
                'QUALITY_PER_DOLLAR': Gauge('pdf_rag_quality_per_dollar', 'Quality (adherence) per dollar spent'),
                'SCORE_LLM_EVALUATOR': Gauge('pdf_rag_llm_evaluator', 'Score from a llm evaluator'),

            }
            
            _METRICS_INITIALIZED = True
            
            # Display registered metrics
            registered_metrics = [
                collector._name for collector in REGISTRY._collector_to_names.keys() 
                if hasattr(collector, '_name') and 'pdf_rag' in collector._name
            ]
            print(f"📊 Registered PDF RAG metrics: {registered_metrics}")
            print(f"📊 Total PDF RAG metrics created: {len(_METRICS)}")
            
        except ValueError as e:
            # Metrics might already exist - try to get them from registry
            print(f"⚠️ Some metrics may already exist: {e}")
            _METRICS_INITIALIZED = True
    
    return _METRICS


def debug_prometheus_endpoint():
    """Debug function to test Prometheus endpoint directly."""
    try:
        import requests
        response = requests.get('http://localhost:8080/metrics', timeout=5)
        print(f"📊 Prometheus endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            content = response.text
            pdf_rag_lines = [line for line in content.split('\n') if 'pdf_rag' in line and not line.startswith('#')]
            print(f"📊 Found {len(pdf_rag_lines)} PDF RAG metric lines")
            
            if pdf_rag_lines:
                print("📊 Sample PDF RAG metrics:")
                for line in pdf_rag_lines[:10]:  # Show first 10
                    print(f"   {line}")
                
                # Count different metric types
                counter_metrics = [line for line in pdf_rag_lines if 'total' in line]
                histogram_metrics = [line for line in pdf_rag_lines if ('_bucket' in line or '_count' in line or '_sum' in line)]
                gauge_metrics = [line for line in pdf_rag_lines if 'avg' in line or 'percentage' in line or 'per_dollar' in line]
                
                print(f"📊 Metric breakdown: {len(counter_metrics)} counters, {len(histogram_metrics)} histogram components, {len(gauge_metrics)} gauges")
            else:
                print("❌ No PDF RAG metrics found!")
                
        return response.status_code == 200 and len(pdf_rag_lines) > 0
        
    except ImportError:
        print("⚠️ requests module not available for endpoint testing")
        return None
    except Exception as e:
        print(f"❌ Error accessing Prometheus endpoint: {e}")
        return False


class Logging:
    """Handles training data persistence for model improvement."""

    @staticmethod
    def save_training_data(query: str, response: str, satisfaction_score: float, context_adherence: float) -> None:
        """
        Save query-response pairs with quality metrics for future training.
        
        Args:
            query: User input query
            response: System generated response
            satisfaction_score: User satisfaction score (0-5)
            context_adherence: Context adherence score (0-1)
        """
        try:
            # Calculate composite quality score
            normalized_satisfaction = satisfaction_score / 5.0
            output_quality = (normalized_satisfaction + context_adherence) / 2
            
            training_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "satisfaction_score": satisfaction_score,
                "context_adherence": context_adherence,
                "output_quality": output_quality
            }
            
            with open("training_data.jsonl", 'a', encoding='utf-8') as f:
                f.write(json.dumps(training_entry, ensure_ascii=False) + '\n')
            
            print(f"💾 Training data saved: Quality={output_quality:.3f} "
                  f"(Satisfaction={satisfaction_score}/5, Adherence={context_adherence:.3f})")
        except Exception as e:
            print(f"❌ Error saving training data: {e}")


class Monitoring:
    """Manages Prometheus metrics collection and monitoring server."""


    @staticmethod
    def clear_prometheus_metrics() -> None:
        """Clear existing PDF RAG metrics to prevent duplicate registration errors."""
        try:
            collectors_to_remove = [
                collector for collector in REGISTRY._collector_to_names.keys()
                if hasattr(collector, '_name') and collector._name.startswith('pdf_rag_')
            ]
            
            for collector in collectors_to_remove:
                try:
                    REGISTRY.unregister(collector)
                except KeyError:
                    pass  # Already removed
            
            if collectors_to_remove:
                print(f"🧹 Cleared {len(collectors_to_remove)} existing PDF RAG metrics")
                
            # Reset the initialization flag so metrics can be re-created
            global _METRICS_INITIALIZED
            _METRICS_INITIALIZED = False
            
        except Exception as e:
            print(f"❌ Error clearing metrics: {e}")

    def create_metrics_app(self) -> Flask:
        """
        Create Flask app for Prometheus metrics endpoint.
        
        Returns:
            Flask app configured with /metrics and /health endpoints
        """
        app = Flask(__name__)
        
        @app.route('/metrics')
        def metrics():
            return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
        
        @app.route('/health')
        def health():
            return {'status': 'healthy', 'service': 'pdf-rag-app'}
        
        return app

    def run_metrics_server(self) -> None:
        """Run the Prometheus metrics server on port 8080."""
        try:
            # Check if port is already in use
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(('localhost', 8080)) == 0:
                    print("📊 Prometheus metrics server already running on port 8080")
                    return
            
            metrics_app = self.create_metrics_app()
            print("✅ Prometheus metrics server started on port 8080")
            print("📊 Metrics available at: http://localhost:8080/metrics")
            metrics_app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
        except OSError as e:
            if "Address already in use" in str(e):
                print("📊 Prometheus metrics server already running on port 8080")
            else:
                print(f"❌ Error starting metrics server: {e}")
                raise
        except Exception as e:
            print(f"❌ Unexpected error in metrics server: {e}")

    @staticmethod
    def log_metrics_to_prometheus(metrics_data: dict, satisfaction_score: float = None) -> None:
        """
        Log all metrics to Prometheus.
        
        Args:
            metrics_data: Dictionary containing metrics (tokens, latency, cost, etc.)
            satisfaction_score: Optional user satisfaction score
        """
        try:
            # Get metrics instance - this will create them if they don't exist
            metrics = get_or_create_metrics()
            
            # Verify metrics_data has required fields
            required_fields = ['input_tokens', 'output_tokens', 'total_cost', 'total_latency', 'combined_adherence']
            missing_fields = [field for field in required_fields if field not in metrics_data]
            if missing_fields:
                print(f"⚠️ Missing required fields in metrics_data: {missing_fields}")
                return

            # Update counters and histograms (individual query metrics)
            metrics['QUERY_COUNTER'].inc()
            metrics['TOKEN_USAGE'].labels(token_type='input').inc(metrics_data['input_tokens'])
            metrics['TOKEN_USAGE'].labels(token_type='output').inc(metrics_data['output_tokens'])
            metrics['COST_TOTAL'].inc(metrics_data['total_cost'])
            
            # Record histograms
            metrics['QUERY_DURATION'].observe(metrics_data['total_latency'])
            metrics['CONTEXT_ADHERENCE'].observe(metrics_data['combined_adherence'])
            metrics['RETRIEVAL_TIME'].observe(metrics_data.get('retrieval_time', 0))
            metrics['LLM_TIME'].observe(metrics_data.get('llm_time', 0))
            
          
            
            print(f"📊 Individual metrics logged to Prometheus: {metrics_data['input_tokens']} input tokens, "
                  f"{metrics_data['output_tokens']} output tokens, {metrics_data['total_latency']:.3f}s latency")
            
        except Exception as e:
            print(f"❌ Error logging individual metrics to Prometheus: {e}")
            # Print the metrics_data for debugging
            print(f"Debug - metrics_data: {metrics_data}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def log_session_metrics_to_prometheus(session_data: dict) -> None:
        """
        Log session-level aggregate metrics to Prometheus.
        
        Args:
            session_data: Dictionary containing session averages and statistics
        """
        try:
            # Get metrics instance
            metrics = get_or_create_metrics()
            
            print(f"🔄 Attempting to log {len(session_data)} session metrics...")
            
            # Update session-level gauge metrics
            metrics_updated = 0
            
            if 'tokens_query' in session_data:
                metrics['TOKENS_PER_QUERY'].set(session_data['tokens_query'])
                metrics_updated += 1
            
            if 'context_adherence_query' in session_data:
                metrics['CONTEXT_ADHERENCE_PER_QUERY'].set(session_data['context_adherence_query'])
                metrics_updated += 1
            
            
            if 'carbon_footprint' in session_data:
                metrics['CARBON_FOOTPRINT'].set(session_data['carbon_footprint'])
                metrics_updated += 1
            
            if 'user_satisfaction' in session_data:
                metrics['USER_SATISFACTION'].set(session_data['user_satisfaction'])
                metrics_updated += 1
            
            if 'cost_per_query' in session_data:
                metrics['COST_PER_QUERY'].set(session_data['cost_per_query'])
                metrics_updated += 1
            
            if 'energy_per_query' in session_data:
                metrics['ENERGY_PER_QUERY'].set(session_data['energy_per_query'])
                metrics_updated += 1
            
            # Performance analysis metrics
            if 'satisfaction_per_dollar' in session_data:
                metrics['SATISFACTION_PER_DOLLAR'].set(session_data['satisfaction_per_dollar'])
                metrics_updated += 1
            
            if 'quality_per_dollar' in session_data:
                metrics['QUALITY_PER_DOLLAR'].set(session_data['quality_per_dollar'])
                metrics_updated += 1

            if 'llm_score' in session_data:
                metrics['SCORE_LLM_EVALUATOR'].set(session_data['llm_score'])
                metrics_updated += 1
            
            if 'latency_query' in session_data:
                metrics['LATENCY_PER_QUERY'].set(session_data['latency_query'])
                metrics_updated += 1

            print(f"📊 Session metrics logged to Prometheus: {metrics_updated}/{len(session_data)} metrics updated successfully")
            
            # Debug session data
            print(f"🔍 Session data received: {list(session_data.keys())}")
            
        except Exception as e:
            print(f"❌ Error logging session metrics to Prometheus: {e}")
            print(f"🔍 Session data keys: {list(session_data.keys()) if session_data else 'None'}")
            import traceback
            traceback.print_exc()


class Metrics:
    """Provides utilities for calculating context adherence, token counts, and costs."""
    
    # Common stopwords for context adherence calculation
    _STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    # GPT-4o pricing (as of 2024)
    _PRICING = {
        'input_cost_per_1k': 0.005,   # $0.005 per 1K input tokens
        'output_cost_per_1k': 0.015   # $0.015 per 1K output tokens
    }


    @staticmethod

    def debug_metrics():
        """Debug function to check what metrics are registered."""
        from prometheus_client import REGISTRY
        all_metrics = []
        for collector in REGISTRY._collector_to_names.keys():
            if hasattr(collector, '_name'):
                all_metrics.append(collector._name)
        
        pdf_rag_metrics = [m for m in all_metrics if 'pdf_rag' in m]
        print(f"🔍 All PDF RAG metrics in registry: {pdf_rag_metrics}")
        return pdf_rag_metrics

    @staticmethod
    def enhanced_retriever_tool(query: str, retriever) -> Tuple[str, List[str]]:
        """
        Enhanced retriever that returns formatted result and raw documents.
        
        Args:
            query: Search query
            retriever: Document retriever instance
            
        Returns:
            Tuple of (formatted_results, raw_document_contents)
        """
        try:
            docs = retriever.invoke(query)
            
            if not docs:
                return "I found no relevant information in the document.", []
            
            formatted_results = []
            raw_docs = []
            
            for i, doc in enumerate(docs):
                formatted_results.append(f"Document {i+1}:\n{doc.page_content}")
                raw_docs.append(doc.page_content)
            
            return "\n\n".join(formatted_results), raw_docs
        except Exception as e:
            print(f"❌ Error in enhanced_retriever_tool: {e}")
            return f"Error retrieving documents: {e}", []

    @classmethod
    def calculate_context_adherence(cls, response: str, retrieved_docs: List[str]) -> float:
        """
        Calculate context adherence score based on word overlap.
        
        Args:
            response: Generated response text
            retrieved_docs: List of retrieved document contents
            
        Returns:
            Adherence score between 0 and 1
        """
        try:
            if not response or not retrieved_docs:
                return 0.0
            
            # Prepare text for comparison
            combined_context = " ".join(retrieved_docs).lower()
            response_lower = response.lower()
            
            # Extract meaningful words (length > 3, not stopwords)
            response_words = {
                word.strip('.,!?;:"()[]') for word in response_lower.split()
                if len(word) > 3 and word not in cls._STOPWORDS
            }
            
            if not response_words:
                return 0.0
            
            # Count matching words
            matching_words = sum(1 for word in response_words if word in combined_context)
            
            return min(matching_words / len(response_words), 1.0)
        except Exception as e:
            print(f"❌ Error calculating context adherence: {e}")
            return 0.0

    @staticmethod
    def calculate_semantic_overlap(response: str, retrieved_docs: List[str]) -> float:
        """
        Calculate semantic overlap using n-gram analysis.
        
        Args:
            response: Generated response text
            retrieved_docs: List of retrieved document contents
            
        Returns:
            Semantic overlap score between 0 and 1
        """
        try:
            if not response or not retrieved_docs:
                return 0.0
            
            combined_context = " ".join(retrieved_docs).lower()
            words = response.lower().split()
            
            # Generate 2-grams and 3-grams
            ngrams = set()
            for i in range(len(words) - 1):
                ngrams.add(f"{words[i]} {words[i+1]}")
            for i in range(len(words) - 2):
                ngrams.add(f"{words[i]} {words[i+1]} {words[i+2]}")
            
            if not ngrams:
                return 0.0
            
            # Count n-gram matches
            matches = sum(1 for ngram in ngrams if ngram in combined_context)
            
            return min(matches / len(ngrams), 1.0)
        except Exception as e:
            print(f"❌ Error calculating semantic overlap: {e}")
            return 0.0

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4o") -> int:
        """
        Count tokens in text for the specified model.
        
        Args:
            text: Input text to tokenize
            model: Model name for tokenization
            
        Returns:
            Number of tokens
        """
        try:
            if not text:
                return 0
            
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback for newer models
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
        except Exception as e:
            print(f"❌ Error counting tokens: {e}")
            # Rough estimate: ~4 characters per token
            return len(text) // 4

    @classmethod
    def calculate_cost(cls, input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> Tuple[float, float, float]:
        """
        Calculate cost based on OpenAI pricing.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name (currently uses GPT-4o pricing)
            
        Returns:
            Tuple of (total_cost, input_cost, output_cost)
        """
        try:
            input_cost = (input_tokens / 1000) * cls._PRICING['input_cost_per_1k']
            output_cost = (output_tokens / 1000) * cls._PRICING['output_cost_per_1k']
            
            return input_cost + output_cost, input_cost, output_cost
        except Exception as e:
            print(f"❌ Error calculating cost: {e}")
            return 0.0, 0.0, 0.0


class LatencyTracker:
    """Tracks execution timing across different processing steps."""
    
    def __init__(self):
        self.start_time: float = None
        self.timings: dict = {}
    
    def start(self) -> None:
        """Initialize timing tracking."""
        self.start_time = time.perf_counter()
        self.timings = {}
    
    def log_step(self, step_name: str) -> None:
        """
        Log timestamp for a processing step.
        
        Args:
            step_name: Name of the processing step
        """
        try:
            current_time = time.perf_counter()
            if self.start_time is None:
                self.start_time = current_time
            
            self.timings[step_name] = current_time - self.start_time
        except Exception as e:
            print(f"❌ Error logging step {step_name}: {e}")
    
    def get_duration(self, start_step: str, end_step: str) -> float:
        """
        Get duration between two steps.
        
        Args:
            start_step: Starting step name
            end_step: Ending step name
            
        Returns:
            Duration in seconds, or 0.0 if steps not found
        """
        try:
            if start_step in self.timings and end_step in self.timings:
                return self.timings[end_step] - self.timings[start_step]
            return 0.0
        except Exception as e:
            print(f"❌ Error getting duration: {e}")
            return 0.0
    
    def get_total_time(self) -> float:
        """
        Get total elapsed time since start.
        
        Returns:
            Total time in seconds, or 0.0 if no timings recorded
        """
        try:
            return max(self.timings.values()) if self.timings else 0.0
        except Exception as e:
            print(f"❌ Error getting total time: {e}")
            return 0.0


# Test function to verify metrics are working
def test_metrics():
    """Test function to verify metrics are properly initialized."""
    try:
        test_data = {
            'input_tokens': 10,
            'output_tokens': 20,
            'total_cost': 0.001,
            'total_latency': 1.5,
            'combined_adherence': 0.8,
            'retrieval_time': 0.3,
            'llm_time': 1.2
        }
        
        print("🧪 Testing individual metrics logging...")
        Monitoring.log_metrics_to_prometheus(test_data, 4)
        
        # Test session metrics
        session_test_data = {
            'avg_tokens_per_query': 15.0,
            'avg_context_adherence': 0.8,
            'avg_latency': 1.5,
            'avg_carbon_per_query': 5.0,
            'avg_user_satisfaction': 4.0,
            'avg_cost_per_query': 0.001,
            'avg_energy_per_query': 0.0001,
            'satisfaction_per_dollar': 4000.0,
            'quality_per_dollar': 800.0,
            'excellent_answers_pct': 80.0,
            'good_answers_pct': 20.0,
            'poor_answers_pct': 0.0
        }
        
        print("🧪 Testing session metrics logging...")
        Monitoring.log_session_metrics_to_prometheus(session_test_data)
        
        print("✅ All metrics tests successful!")
        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Initialize metrics when module is imported (but after clearing if needed)
def init_metrics():
    """Initialize metrics - call this after clearing if needed."""
    return get_or_create_metrics()


if __name__ == "__main__":
    # Run test when module is executed directly
    print("🧪 Running comprehensive metrics test...")
    test_metrics()
    
    # Test endpoint if possible
    print("\n🔍 Testing Prometheus endpoint...")
    debug_prometheus_endpoint()