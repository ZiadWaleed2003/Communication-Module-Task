from CommunicationModule.blackboard import Blackboard
from clients import get_llm_client


class Agent:
    """
    Base agent that communicates through the Blackboard using LLM calls
    """
    def __init__(self, agent_id: str, role: str, system_prompt: str):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.step_count = 0
        self.llm = get_llm_client()
    
    def generate_response(self, problem: str, conversation_history: str) -> str:
        """
        Generate a response using LLM (OpenAI GPT-4)
        For demo purposes, we'll simulate LLM responses
        """

        response = self.llm.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Problem: {problem}\n\n{conversation_history}\n\nProvide your response:"}
            ]
        )
        return response.choices[0].message.content
        
    
    
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