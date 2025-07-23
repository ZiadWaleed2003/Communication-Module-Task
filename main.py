import json
import time
import random
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import openai
import os

# Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class Message:
    """Represents a message on the Blackboard"""
    id: str
    agent_id: str
    agent_role: str
    content: str
    timestamp: datetime
    message_type: str = "response"  # "response", "question", "solution", "analysis"
    metadata: Dict[str, Any] = field(default_factory=dict)

class Blackboard:
    """
    Blackboard communication system where all agents can read and write messages
    """
    def __init__(self):
        self.messages: List[Message] = []
        self.message_counter = 0
    
    def post_message(self, agent_id: str, agent_role: str, content: str, 
                    message_type: str = "response", metadata: Dict = None) -> str:
        """Post a message to the blackboard"""
        self.message_counter += 1
        message_id = f"msg_{self.message_counter:04d}"
        
        message = Message(
            id=message_id,
            agent_id=agent_id,
            agent_role=agent_role,
            content=content,
            timestamp=datetime.now(),
            message_type=message_type,
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        return message_id
    
    def get_all_messages(self) -> List[Message]:
        """Get all messages from the blackboard"""
        return self.messages.copy()
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history for LLM prompts"""
        if not self.messages:
            return "No previous messages."
        
        history = "=== CONVERSATION HISTORY ===\n"
        for msg in self.messages:
            timestamp = msg.timestamp.strftime("%H:%M:%S")
            history += f"[{timestamp}] {msg.agent_role} ({msg.agent_id}):\n{msg.content}\n\n"
        
        return history
    
    def clear(self):
        """Clear all messages from the blackboard"""
        self.messages.clear()
        self.message_counter = 0

class Agent:
    """
    Base agent that communicates through the Blackboard using LLM calls
    """
    def __init__(self, agent_id: str, role: str, system_prompt: str):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.step_count = 0
    
    def generate_response(self, problem: str, conversation_history: str) -> str:
        """
        Generate a response using LLM (OpenAI GPT-4)
        For demo purposes, we'll simulate LLM responses
        """
        # In real implementation, you would call OpenAI API:
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": self.system_prompt},
        #         {"role": "user", "content": f"Problem: {problem}\n\n{conversation_history}\n\nProvide your response:"}
        #     ]
        # )
        # return response.choices[0].message.content
        
        # SIMULATED LLM RESPONSES (replace with real API calls)
        return self._simulate_llm_response(problem, conversation_history)
    
    def _simulate_llm_response(self, problem: str, conversation_history: str) -> str:
        """Simulate LLM responses for demo purposes"""
        self.step_count += 1
        
        # Simple simulation based on role
        if "analyst" in self.role.lower():
            responses = [
                f"I've analyzed the problem and identified key factors: {random.choice(['data quality', 'algorithmic complexity', 'resource constraints'])}. Let me break this down further.",
                f"From my analysis, this appears to be a {random.choice(['optimization', 'classification', 'regression'])} problem. Here are my observations...",
                f"I notice some patterns in the data that suggest we should consider {random.choice(['feature engineering', 'model selection', 'validation strategy'])}."
            ]
        elif "coordinator" in self.role.lower():
            responses = [
                "Let me synthesize what we've discussed so far and propose next steps for the team.",
                "Based on everyone's input, I suggest we focus on the most promising approach and divide the remaining tasks.",
                f"Good progress team! Let's now focus on {random.choice(['implementation', 'validation', 'optimization'])} phase."
            ]
        elif "specialist" in self.role.lower():
            responses = [
                f"From my domain expertise, I recommend using {random.choice(['ensemble methods', 'deep learning', 'statistical approaches'])} for this specific case.",
                "I've seen similar problems before. The key insight here is to handle the edge cases carefully.",
                f"My specialized knowledge suggests that {random.choice(['cross-validation', 'feature selection', 'hyperparameter tuning'])} will be critical here."
            ]
        else:
            responses = [
                "I agree with the previous analysis and would like to add my perspective on this problem.",
                "Let me contribute by focusing on the implementation details we need to consider.",
                "Here's my take on the problem based on what I've heard so far..."
            ]
        
        return f"{random.choice(responses)} [Step {self.step_count}]"
    
    def act(self, blackboard: Blackboard, problem: str) -> str:
        """
        Agent's main action: read blackboard, generate response, post to blackboard
        """
        # Read conversation history
        conversation_history = blackboard.get_conversation_history()
        
        # Generate response using LLM
        response = self.generate_response(problem, conversation_history)
        
        # Post response to blackboard
        message_id = blackboard.post_message(
            agent_id=self.agent_id,
            agent_role=self.role,
            content=response,
            message_type="response"
        )
        
        return message_id

class CollaBArenSimulation:
    """
    Main simulation class that orchestrates the multi-agent collaboration
    """
    def __init__(self):
        self.blackboard = Blackboard()
        self.agents: List[Agent] = []
        self.datasets = self._load_sample_datasets()
    
    def _load_sample_datasets(self) -> List[Dict]:
        """
        Load sample problems from m500 dataset (simulated for demo)
        In real implementation, this would load actual dataset files
        """
        return [
            {
                "id": "math_001",
                "problem": "A company has 3 factories producing widgets. Factory A produces 100 widgets/day, Factory B produces 150 widgets/day, and Factory C produces 200 widgets/day. If the company needs to produce 10,000 widgets in the minimum number of days, how should they allocate production?",
                "domain": "optimization",
                "difficulty": "medium"
            },
            {
                "id": "logic_002", 
                "problem": "In a tournament, each team plays every other team exactly once. If there are 8 teams total and each game has exactly one winner, what's the minimum number of games needed to determine a clear overall winner?",
                "domain": "combinatorics",
                "difficulty": "hard"
            },
            {
                "id": "analysis_003",
                "problem": "A data scientist has a dataset with 10,000 samples and 50 features. The target variable is continuous. They want to build a predictive model but are concerned about overfitting. What approach should they take?",
                "domain": "machine_learning", 
                "difficulty": "medium"
            }
        ]
    
    def setup_agents(self) -> None:
        """Initialize the agent team with different roles"""
        agent_configs = [
            {
                "id": "analyst_01",
                "role": "Problem Analyst",
                "system_prompt": "You are a Problem Analyst. Your role is to break down complex problems, identify key components, and provide structured analysis. Be methodical and thorough."
            },
            {
                "id": "coordinator_01", 
                "role": "Team Coordinator",
                "system_prompt": "You are a Team Coordinator. Your role is to synthesize team input, facilitate collaboration, and guide the problem-solving process. Focus on organizing ideas and next steps."
            },
            {
                "id": "specialist_01",
                "role": "Domain Specialist", 
                "system_prompt": "You are a Domain Specialist with deep expertise. Your role is to provide specialized insights, identify domain-specific approaches, and suggest advanced techniques."
            },
            {
                "id": "implementer_01",
                "role": "Solution Implementer",
                "system_prompt": "You are a Solution Implementer. Your role is to focus on practical implementation, identify potential issues, and propose concrete next steps."
            }
        ]
        
        self.agents = [
            Agent(config["id"], config["role"], config["system_prompt"])
            for config in agent_configs
        ]
    
    def run_simulation(self, problem_data: Dict, max_steps: int = 6) -> Dict:
        """
        Run a single simulation scenario
        """
        print(f"\n{'='*80}")
        print(f"STARTING SIMULATION: {problem_data['id']}")
        print(f"Domain: {problem_data['domain']} | Difficulty: {problem_data['difficulty']}")
        print(f"{'='*80}")
        print(f"PROBLEM: {problem_data['problem']}")
        print(f"{'='*80}\n")
        
        # Clear blackboard for new simulation
        self.blackboard.clear()
        
        # Initialize problem on blackboard
        self.blackboard.post_message(
            agent_id="system",
            agent_role="System",
            content=f"PROBLEM TO SOLVE: {problem_data['problem']}",
            message_type="problem_statement"
        )
        
        # Run collaboration steps
        for step in range(max_steps):
            print(f"\n--- COLLABORATION STEP {step + 1} ---")
            
            # Each agent takes a turn
            for agent in self.agents:
                print(f"\n🤖 {agent.role} ({agent.agent_id}) is thinking...")
                
                # Agent reads blackboard and generates response
                message_id = agent.act(self.blackboard, problem_data['problem'])
                
                # Display the response
                messages = self.blackboard.get_all_messages()
                latest_msg = messages[-1]
                print(f"💬 Response: {latest_msg.content}")
                
                # Add small delay for readability
                time.sleep(0.5)
            
            print(f"\n--- END OF STEP {step + 1} ---")
        
        # Return simulation results
        return {
            "problem_id": problem_data['id'],
            "total_messages": len(self.blackboard.get_all_messages()),
            "final_state": self.blackboard.get_conversation_history(),
            "agents_participated": len(self.agents)
        }
    
    def run_all_simulations(self) -> List[Dict]:
        """
        Run simulations for all problems in the dataset
        """
        print("🚀 STARTING COLLAB ARENA SIMULATION")
        print("Testing Blackboard Communication Protocol with LLM-Powered Agents")
        print(f"Dataset Size: {len(self.datasets)} problems")
        print(f"Agent Team Size: {len(self.agents)} agents")
        
        results = []
        
        for i, problem in enumerate(self.datasets):
            print(f"\n\n📊 SIMULATION {i+1}/{len(self.datasets)}")
            result = self.run_simulation(problem)
            results.append(result)
            
            # Brief pause between simulations
            if i < len(self.datasets) - 1:
                print(f"\n⏳ Preparing next simulation...")
                time.sleep(2)
        
        return results
    
    def print_final_summary(self, results: List[Dict]) -> None:
        """Print summary of all simulation results"""
        print(f"\n\n{'='*80}")
        print("🎯 SIMULATION SUMMARY")
        print(f"{'='*80}")
        
        total_messages = sum(r['total_messages'] for r in results)
        avg_messages = total_messages / len(results) if results else 0
        
        print(f"✅ Completed Simulations: {len(results)}")
        print(f"📝 Total Messages Generated: {total_messages}")
        print(f"📊 Average Messages per Problem: {avg_messages:.1f}")
        print(f"🤖 Agents in Team: {results[0]['agents_participated'] if results else 0}")
        
        print(f"\n📋 Problem Breakdown:")
        for result in results:
            print(f"  • {result['problem_id']}: {result['total_messages']} messages")
        
        print(f"\n🏆 Simulation completed successfully!")
        print("The Blackboard communication protocol is working correctly.")

def main():
    """
    Main function to run the CollaB Arena simulation
    """
    # Create simulation instance
    simulation = CollaBArenSimulation()
    
    # Setup agent team
    simulation.setup_agents()
    
    # Run all simulations
    results = simulation.run_all_simulations()
    
    # Print final summary
    simulation.print_final_summary(results)

if __name__ == "__main__":
    main()