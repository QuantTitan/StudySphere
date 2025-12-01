import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any
import json
from datetime import datetime

class AgentEvaluator:
    """LLM-as-judge evaluation of agent responses"""
    
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.7
        )
        self.evaluations = []
    
    EVALUATION_PROMPT = """You are an educational assessment expert. Evaluate the following tutor response on a scale of 1-5 for:

1. **Accuracy** (Does it match course materials? No hallucinations?)
2. **Relevance** (Does it directly answer the question?)
3. **Clarity** (Is it well-structured and easy to understand?)
4. **Citation Quality** (Are sources properly cited?)
5. **Pedagogical Value** (Does it help students learn?)

**Question:** {question}

**Response:** {response}

**Sources Referenced:** {sources}

Provide a JSON response with:
{{"accuracy": score, "relevance": score, "clarity": score, "citations": score, "pedagogical": score, "overall": avg_score, "feedback": "brief feedback"}}"""

    def evaluate_response(self, question: str, response: str, sources: str) -> Dict[str, Any]:
        """Use LLM-as-judge to evaluate response"""
        prompt = self.EVALUATION_PROMPT.format(
            question=question,
            response=response,
            sources=sources
        )
        
        evaluation_text = self.llm.predict(prompt)
        
        try:
            # Parse JSON response
            import json
            evaluation = json.loads(evaluation_text)
        except:
            evaluation = {"error": "Could not parse evaluation", "raw": evaluation_text}
        
        self.evaluations.append({
            "timestamp": datetime.now().isoformat(),
            "question": question[:100],
            "evaluation": evaluation
        })
        
        return evaluation
    
    def batch_evaluate(self, test_cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Evaluate multiple test cases"""
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"Evaluating test case {i}/{len(test_cases)}...")
            evaluation = self.evaluate_response(
                case["question"],
                case["response"],
                case.get("sources", "")
            )
            results.append(evaluation)
        return results
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Summarize all evaluations"""
        if not self.evaluations:
            return {"status": "No evaluations yet"}
        
        avg_scores = {
            "accuracy": 0,
            "relevance": 0,
            "clarity": 0,
            "citations": 0,
            "pedagogical": 0,
            "overall": 0
        }
        
        count = 0
        for eval_entry in self.evaluations:
            if "evaluation" in eval_entry and "error" not in eval_entry["evaluation"]:
                eval_data = eval_entry["evaluation"]
                for key in avg_scores:
                    if key in eval_data:
                        avg_scores[key] += eval_data[key]
                count += 1
        
        if count > 0:
            for key in avg_scores:
                avg_scores[key] /= count
        
        return {
            "total_evaluations": len(self.evaluations),
            "avg_scores": avg_scores,
            "evaluations": self.evaluations
        }
    
    def export_evaluations(self, filepath: str = "agent_evaluations.json"):
        """Export evaluation results"""
        with open(filepath, 'w') as f:
            json.dump(self.get_evaluation_summary(), f, indent=2)
        print(f"Evaluations exported to {filepath}")

# Test prompts for accuracy evaluation
TEST_PROMPTS = [
    {
        "question": "What are the main steps in the Kalman Filter algorithm?",
        "expected_topics": ["prediction", "update", "covariance", "state estimation"],
        "category": "Concept Understanding"
    },
    {
        "question": "How does reinforcement learning differ from supervised learning?",
        "expected_topics": ["reward", "agent", "policy", "exploration"],
        "category": "Comparative Analysis"
    },
    {
        "question": "Explain the off-policy learning approach in RL.",
        "expected_topics": ["behavior policy", "target policy", "importance sampling"],
        "category": "Advanced Concept"
    },
    {
        "question": "What is the role of the covariance matrix in Kalman Filtering?",
        "expected_topics": ["uncertainty", "error", "matrix", "estimation"],
        "category": "Technical Details"
    },
    {
        "question": "Can you provide an example of Kalman Filter application?",
        "expected_topics": ["tracking", "prediction", "navigation", "filtering"],
        "category": "Application"
    }
]