import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from openai import OpenAI
import tiktoken
from typing import List, Dict, Tuple
import time
from dotenv import load_dotenv

load_dotenv()

minimum_data_points = 10

class RLHFTrainingPipeline:
    """Complete RLHF training pipeline for PDF RAG system."""
    
    def __init__(self):
        self.client = OpenAI()
        self.reward_model = None
        self.high_quality_data = None
        self.encoding = tiktoken.encoding_for_model("gpt-4o")
        
    def load_training_data(self, file_path: str = "training_data.jsonl") -> List[Dict]:
        """Load training data from JSONL file."""
        training_data = []
        
        if not os.path.exists(file_path):
            print(f"❌ Training data file {file_path} not found!")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        training_data.append(json.loads(line.strip()))
            
            print(f"📊 Loaded {len(training_data)} training samples")
            return training_data
        
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return []
    
    def filter_high_quality_data(self, training_data: List[Dict], percentile: int = 80) -> List[Dict]:
        """Filter data to keep only high-quality samples (80th percentile and above)."""
        
        if not training_data:
            print("❌ No training data to filter!")
            return []
        
        # Extract output quality scores
        quality_scores = [sample['output_quality'] for sample in training_data]
        
        # Calculate 80th percentile threshold
        threshold = np.percentile(quality_scores, percentile)
        
        # Filter high-quality samples
        high_quality_samples = [
            sample for sample in training_data 
            if sample['output_quality'] >= threshold
        ]
        
        print(f"📈 Quality threshold (80th percentile): {threshold:.3f}")
        print(f"🔥 High-quality samples: {len(high_quality_samples)}/{len(training_data)} ({len(high_quality_samples)/len(training_data)*100:.1f}%)")
        
        # Save filtered data
        self.save_filtered_data(high_quality_samples, f"high_quality_data_p{percentile}.jsonl")
        
        return high_quality_samples
    
    def save_filtered_data(self, data: List[Dict], filename: str):
        """Save filtered data to file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for sample in data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"💾 Saved {len(data)} high-quality samples to {filename}")
        
        except Exception as e:
            print(f"❌ Error saving filtered data: {e}")
    
    def extract_features(self, query: str, response: str) -> np.ndarray:
        """Extract features from query-response pairs for reward model."""
        
        # Basic text features
        query_length = len(query)
        response_length = len(response)
        query_tokens = len(self.encoding.encode(query))
        response_tokens = len(self.encoding.encode(response))
        
        # Lexical features
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        word_overlap = len(query_words.intersection(response_words)) / max(len(query_words), 1)
        
        # Response quality indicators
        has_citations = any(keyword in response.lower() for keyword in ['document', 'page', 'section'])
        question_marks = response.count('?')
        exclamation_marks = response.count('!')
        
        # Readability features
        sentences = response.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        features = np.array([
            query_length,
            response_length,
            query_tokens,
            response_tokens,
            word_overlap,
            has_citations,
            question_marks,
            exclamation_marks,
            avg_sentence_length,
            len(sentences)
        ])
        
        return features
    
    def train_reward_model(self, high_quality_data: List[Dict]) -> Dict:
        """Phase 3: Train reward model to predict human satisfaction scores."""
        
        if len(high_quality_data) < minimum_data_points /2:
            print("❌ Not enough high-quality data for reward model training!")
            return {}
        
        print("🤖 Phase 3: Training Reward Model...")
        
        # Prepare features and targets
        X = []
        y = []
        
        for sample in high_quality_data:
            features = self.extract_features(sample['query'], sample['response'])
            X.append(features)
            y.append(sample['satisfaction_score'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100000, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        best_model = None
        best_score = -float('inf')
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mse': mse,
                'r2': r2,
                'model': model
            }
            
            print(f"  {name}: MSE={mse:.3f}, R²={r2:.3f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                self.reward_model = model
        
        # Save best reward model
        model_filename = f"reward_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(best_model, model_filename)
        
        print(f"💾 Best reward model saved as {model_filename}")
        print(f"✅ Reward model training complete! Best R²: {best_score:.3f}")
        
        return results
    
    def predict_satisfaction(self, query: str, response: str) -> float:
        """Use trained reward model to predict satisfaction score."""
        if self.reward_model is None:
            print("❌ No reward model loaded!")
            return 0.0
        
        features = self.extract_features(query, response).reshape(1, -1)
        predicted_score = self.reward_model.predict(features)[0]
        
        # Clamp to valid range [0, 5]
        return np.clip(predicted_score, 0, 5)
    
    def prepare_fine_tuning_data(self, high_quality_data: List[Dict]) -> str:
        """Prepare data in OpenAI fine-tuning format."""
        
        fine_tuning_data = []
        
        for sample in high_quality_data:
            # Format for instruction following
            training_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an intelligent AI assistant who answers questions about PDF documents. Always provide accurate, well-cited responses based on the retrieved context."
                    },
                    {
                        "role": "user",
                        "content": sample['query']
                    },
                    {
                        "role": "assistant",
                        "content": sample['response']
                    }
                ]
            }
            
            fine_tuning_data.append(training_example)
        
        # Save training data
        filename = f"fine_tuning_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(filename, 'w', encoding='utf-8') as f:
            for example in fine_tuning_data:
                f.write(json.dumps(example) + '\n')
        
        print(f"💾 Fine-tuning data saved as {filename}")
        return filename
    
    def fine_tune_model(self, training_file: str, model_name: str = "gpt-4o-2024-08-06") -> str:
        """Phase 4: Fine-tune the main LLM using high-quality data."""
        
        print("🎯 Phase 4: Fine-tuning Main LLM...")
        
        try:
            # Upload training file
            print("📤 Uploading training file...")
            with open(training_file, 'rb') as f:
                training_file_obj = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            print(f"✅ Training file uploaded: {training_file_obj.id}")
            
            # Create fine-tuning job
            print("🚀 Starting fine-tuning job...")
            fine_tuning_job = self.client.fine_tuning.jobs.create(
                training_file=training_file_obj.id,
                model=model_name,
                hyperparameters={
                    "n_epochs": 3,  # Adjust based on your data size
                    "batch_size": 1,
                    "learning_rate_multiplier": 0.1
                }
            )
            
            job_id = fine_tuning_job.id
            print(f"✅ Fine-tuning job created: {job_id}")
            
            # Monitor job progress
            print("⏳ Monitoring fine-tuning progress...")
            while True:
                job_status = self.client.fine_tuning.jobs.retrieve(job_id)
                status = job_status.status
                
                print(f"   Status: {status}")
                
                if status == 'succeeded':
                    model_id = job_status.fine_tuned_model
                    print(f"🎉 Fine-tuning completed! Model ID: {model_id}")
                    
                    # Save model info
                    model_info = {
                        'model_id': model_id,
                        'job_id': job_id,
                        'created_at': datetime.now().isoformat(),
                        'training_file': training_file,
                        'base_model': model_name
                    }
                    
                    with open(f"fine_tuned_model_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                        json.dump(model_info, f, indent=2)
                    
                    return model_id
                
                elif status in ['failed', 'cancelled']:
                    print(f"❌ Fine-tuning failed: {status}")
                    return None
                
                time.sleep(60)  # Check every minute
        
        except Exception as e:
            print(f"❌ Error during fine-tuning: {e}")
            return None
    
    def run_complete_pipeline(self, training_data_file: str = "training_data.jsonl"):
        """Run the complete RLHF pipeline."""
        
        print("🚀 Starting Complete RLHF Pipeline...")
        print("="*50)
        
        # Step 1: Load and filter data
        print("📊 Step 1: Loading and filtering training data...")
        training_data = self.load_training_data(training_data_file)
        
        if len(training_data) < minimum_data_points:
            print("❌ Not enough training data! Need at least 20 samples for reliable training.")
            return None
        
        high_quality_data = self.filter_high_quality_data(training_data, percentile=50)
        
        if len(high_quality_data) < minimum_data_points/2:
            print("❌ Not enough high-quality data after filtering!")
            return None
        
        # Step 2: Train reward model
        print("\n🤖 Step 2: Training reward model...")
        reward_results = self.train_reward_model(high_quality_data)
        
        # Step 3: Prepare fine-tuning data
        print("\n📝 Step 3: Preparing fine-tuning data...")
        fine_tuning_file = self.prepare_fine_tuning_data(high_quality_data)
        
        # Step 4: Fine-tune model
        print("\n🎯 Step 4: Fine-tuning main LLM...")
        fine_tuned_model_id = self.fine_tune_model(fine_tuning_file)
        
        if fine_tuned_model_id:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"   Fine-tuned model: {fine_tuned_model_id}")
            print(f"   Reward model saved and ready for evaluation")
            
            return {
                'fine_tuned_model_id': fine_tuned_model_id,
                'reward_model': self.reward_model,
                'high_quality_samples': len(high_quality_data),
                'reward_model_performance': reward_results
            }
        else:
            print("❌ Pipeline failed during fine-tuning step")
            return None

# Usage example and testing functions
def test_reward_model(pipeline: RLHFTrainingPipeline):
    """Test the trained reward model with sample queries."""
    
    test_cases = [
        {
            "query": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions."
        },
        {
            "query": "Explain neural networks",
            "response": "Networks are good."  # Poor quality response
        }
    ]
    
    print("\n🧪 Testing Reward Model Predictions:")
    for i, test_case in enumerate(test_cases):
        predicted_score = pipeline.predict_satisfaction(test_case['query'], test_case['response'])
        print(f"  Test {i+1}: Predicted satisfaction = {predicted_score:.2f}/5")

def load_and_use_fine_tuned_model(model_id: str):
    """Example of how to load and use the fine-tuned model."""
    
    from openai import OpenAI
    
    client = OpenAI()
    
    # Use the fine-tuned model
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are an intelligent AI assistant who answers questions about PDF documents."},
            {"role": "user", "content": "What is the main topic of this document?"}
        ],
        max_tokens=150,
        temperature=0
    )
    
    return response.choices[0].message.content

# Main execution
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RLHFTrainingPipeline()
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    if results:
        # Test reward model
        test_reward_model(pipeline)
        
        print("\n📋 Training Summary:")
        print(f"   Fine-tuned model: {results['fine_tuned_model_id']}")
        print(f"   High-quality samples used: {results['high_quality_samples']}")
        print(f"   Reward model R²: {results['reward_model_performance']['random_forest']['r2']:.3f}")
        
        # Save results summary
        with open(f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
    else:
        print("❌ Training pipeline failed. Please check your data and try again.")