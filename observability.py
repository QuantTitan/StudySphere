import time
import logging
from datetime import datetime
from typing import Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('studysphere_observability.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ObservabilityTracker:
    """Track metrics: retrieval time, token count, prompt latency"""
    
    def __init__(self):
        self.metrics = {
            "retrieval_times": [],
            "token_counts": [],
            "prompt_latencies": [],
            "queries": []
        }
    
    def log_retrieval(self, query: str, retrieval_time: float, num_results: int):
        """Log RAG retrieval metrics"""
        self.metrics["retrieval_times"].append(retrieval_time)
        logger.info(f"RAG Retrieval | Query: {query[:50]}... | Time: {retrieval_time:.2f}s | Results: {num_results}")
    
    def log_token_count(self, prompt: str, completion: str, token_estimate: int):
        """Log token usage estimate"""
        self.metrics["token_counts"].append(token_estimate)
        logger.info(f"Token Usage | Prompt: {len(prompt)} chars | Completion: {len(completion)} chars | Est. Tokens: {token_estimate}")
    
    def log_prompt_latency(self, agent_name: str, latency: float):
        """Log end-to-end prompt latency"""
        self.metrics["prompt_latencies"].append(latency)
        logger.info(f"Prompt Latency | Agent: {agent_name} | Latency: {latency:.2f}s")
    
    def log_query(self, query: str, agent: str, response: str, total_time: float):
        """Log full query interaction"""
        query_log = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],
            "agent": agent,
            "response_length": len(response),
            "total_time": total_time
        }
        self.metrics["queries"].append(query_log)
        logger.info(f"Query Complete | Agent: {agent} | Time: {total_time:.2f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Return observability summary"""
        if not self.metrics["retrieval_times"]:
            return {"status": "No metrics collected yet"}
        
        return {
            "avg_retrieval_time": sum(self.metrics["retrieval_times"]) / len(self.metrics["retrieval_times"]),
            "max_retrieval_time": max(self.metrics["retrieval_times"]),
            "avg_prompt_latency": sum(self.metrics["prompt_latencies"]) / len(self.metrics["prompt_latencies"]) if self.metrics["prompt_latencies"] else 0,
            "total_queries": len(self.metrics["queries"]),
            "avg_token_count": sum(self.metrics["token_counts"]) / len(self.metrics["token_counts"]) if self.metrics["token_counts"] else 0,
        }
    
    def export_metrics(self, filepath: str = "metrics.json"):
        """Export metrics to JSON for analysis"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics exported to {filepath}")

# Global tracker instance
tracker = ObservabilityTracker()