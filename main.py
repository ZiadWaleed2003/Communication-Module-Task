import time
from typing import Dict, List
from dataclasses import dataclass, field

from src.CommunicationModule.blackboard import Blackboard
from src.agent import Agent

from input_data.data import load_sample_datasets

class CollaBArenSimulation:
    """
    Main simulation class that orchestrates the multi-agent collaboration
    """
    def __init__(self):
        self.blackboard = Blackboard()
        self.agents: List[Agent] = []
        self.datasets = load_sample_datasets()
    
    
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