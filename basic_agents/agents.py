from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
import asyncio
import logging
from datetime import datetime, timedelta
import json
import time

from blackboard_communication import (
    IBlackboard, Message, MessageType, AgentRole, 
    Blackboard, create_blackboard
)

logger = logging.getLogger(__name__)


class AgentState:
    #Current state of an agent
    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.status = "idle"  # idle, working, waiting, error
        self.current_task = None
        self.capabilities = []
        self.load_factor = 0.0  # 0.0 to 1.0
        self.last_activity = datetime.now()
        self.error_count = 0
        self.completed_tasks = 0


class BaseAgent(ABC):
    """
    Base class for all agents in the CollabArena system
    
    Provides common functionality for:
    - Blackboard communication
    - Task management
    - Memory operations
    - Human interaction
    """
    
    def __init__(self, agent_id: str, role: AgentRole, blackboard: IBlackboard,
                 capabilities: List[str] = None):
        self.agent_id = agent_id
        self.role = role
        self.blackboard = blackboard
        self.capabilities = capabilities or []
        self.state = AgentState(agent_id, role)
        self.active = False
        self.message_handlers = {}
        self.task_queue = []
        
        logger.info(f"Agent {self.agent_id} ({self.role.value}) initialized")
    
    def start(self):
        self.active = True
        self.state.status = "idle"
        self._broadcast_status()
        logger.info(f"Agent {self.agent_id} started")
    
    def stop(self):
        self.active = False
        self.state.status = "stopped"
        logger.info(f"Agent {self.agent_id} stopped")
    
    def _handle_incoming_message(self, message: Message):
        try:
            self.state.last_activity = datetime.now()
            
            # Check if message is targeted to this agent
            if message.target_agents and self.agent_id not in message.target_agents:
                return
            
            # Route message to appropriate handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                handler(message)
            else:
                self._default_message_handler(message)
                
        except Exception as e:
            logger.error(f"Agent {self.agent_id} error handling message: {str(e)}")
            self.state.error_count += 1
    
    def _default_message_handler(self, message: Message):
        #Default message handler
        logger.info(f"Agent {self.agent_id} received {message.message_type.value} from {message.sender_id}")
    
    def send_message(self, message_type: MessageType, content: Dict[str, Any],
                    target_agents: List[str] = None, priority: int = 1,
                    requires_response: bool = False, access_level: str = "public") -> bool:
        #Send a message via the blackboard
        try:
            message = Message(
                sender_id=self.agent_id,
                sender_role=self.role,
                message_type=message_type,
                content=content,
                target_agents=target_agents or [],
                priority=priority,
                requires_response=requires_response
            )
            
            return self.blackboard.post_message(message, access_level=access_level)
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} error sending message: {str(e)}")
            return False
    
    def request_task_assignment(self, task_description: str, required_capabilities: List[str] = None):
        content = {
            "task_description": task_description,
            "required_capabilities": required_capabilities or [],
            "agent_capabilities": self.capabilities,
            "current_load": self.state.load_factor
        }
        
        self.send_message(
            MessageType.TASK_REQUEST,
            content,
            requires_response=True
        )
    
    def share_knowledge(self, topic: str, findings: Dict[str, Any], 
                       target_roles: List[AgentRole] = None):
        #Share knowledge with other agents
        content = {
            "topic": topic,
            "findings": findings,
            "source_agent": self.agent_id,
            "confidence": findings.get("confidence", 0.8),
            "timestamp": datetime.now().isoformat()
        }
        
        # Determine access level based on target roles
        access_level = "role_based" if target_roles else "public"
        
        self.send_message(
            MessageType.KNOWLEDGE_SHARE,
            content,
            access_level=access_level
        )
    
    def report_error(self, error_type: str, description: str, severity: str = "medium"):
        content = {
            "error_type": error_type,
            "description": description,
            "severity": severity,
            "agent_state": {
                "status": self.state.status,
                "current_task": self.state.current_task,
                "error_count": self.state.error_count
            }
        }
        
        self.send_message(MessageType.ERROR_REPORT, content, priority=3)
        self.state.error_count += 1
    
    def _broadcast_status(self):
        #Broadcast current status to other agents
        content = {
            "status": self.state.status,
            "load_factor": self.state.load_factor,
            "capabilities": self.capabilities,
            "completed_tasks": self.state.completed_tasks,
            "error_count": self.state.error_count
        }
        
        self.send_message(MessageType.STATUS_UPDATE, content)
    
    def get_shared_knowledge(self, topic: str = None, 
                           since: Optional[datetime] = None) -> List[Message]:
        #Get shared knowledge from the blackboard
        return self.blackboard.read_messages(
            agent_id=self.agent_id,
            agent_role=self.role,
            message_type=MessageType.KNOWLEDGE_SHARE,
            since=since
        )
    
    @abstractmethod
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
      
        pass
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "status": self.state.status,
            "load_factor": self.state.load_factor,
            "capabilities": self.capabilities,
            "completed_tasks": self.state.completed_tasks,
            "error_count": self.state.error_count,
            "last_activity": self.state.last_activity.isoformat()
        }


class ResearcherAgent(BaseAgent):
    """
    Researcher agent specialized in gathering and analyzing information
    """
    
    def __init__(self, agent_id: str, blackboard: IBlackboard):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.RESEARCHER,
            blackboard=blackboard,
            capabilities=["research", "analysis", "data_gathering", "summarization"]
        )
        
        # Set up message handlers
        self.message_handlers[MessageType.TASK_REQUEST] = self._handle_task_request
        
    def _handle_task_request(self, message: Message):
        """Handle task assignment requests"""
        task_info = message.content
        
        # Check if we can handle this task
        required_caps = task_info.get("required_capabilities", [])
        if any(cap in self.capabilities for cap in required_caps):
            self.state.status = "working"
            self.state.current_task = task_info.get("task_description")
            
            # Process the task
            result = self.process_task(task_info)
            
            # Share findings
            if result:
                self.share_knowledge(
                    topic=task_info.get("task_description", "research_task"),
                    findings=result
                )
            
            self.state.status = "idle"
            self.state.completed_tasks += 1
            self._broadcast_status()
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a research task"""
        task_description = task.get('task_description', '')
        logger.info(f"Researcher {self.agent_id} processing: {task_description}")
        
        # Enhanced mathematical problem solving
        if "mathematical" in task_description.lower() or "solve" in task_description.lower():
            return self._solve_mathematical_problem(task_description, task)
        else:
            return self._general_research_task(task_description, task)
    
    def _solve_mathematical_problem(self, problem_text: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to solve mathematical problems"""
        import re  
        
        # Simple pattern matching for common mathematical problems
        solution = "Unable to determine solution"
        confidence = 0.5
        reasoning = ["Problem analysis attempted"]
        
        problem_lower = problem_text.lower()
        
        # Basic arithmetic patterns
        if "+" in problem_text and any(char.isdigit() for char in problem_text):
            # Try to extract and solve simple addition
            numbers = re.findall(r'\d+', problem_text)
            if len(numbers) >= 2:
                try:
                    result = sum(int(num) for num in numbers)
                    solution = f"Sum: {result}"
                    confidence = 0.9
                    reasoning = [f"Added numbers: {' + '.join(numbers)}", f"Result: {result}"]
                except:
                    pass
        
        elif "derivative" in problem_lower:
            # Basic derivative patterns
            if "x^2" in problem_text:
                solution = "2x"
                confidence = 0.85
                reasoning = ["Applied power rule: d/dx(x^n) = n*x^(n-1)", "For x^2: derivative is 2x"]
            elif "x^3" in problem_text:
                solution = "3x^2"
                confidence = 0.85
                reasoning = ["Applied power rule for x^3", "Derivative is 3x^2"]
        
        elif any(op in problem_text for op in ["*", "×", "multiply"]):
            # Try multiplication
            numbers = re.findall(r'\d+', problem_text)
            if len(numbers) >= 2:
                try:
                    result = 1
                    for num in numbers:
                        result *= int(num)
                    solution = f"Product: {result}"
                    confidence = 0.9
                    reasoning = [f"Multiplied numbers: {' × '.join(numbers)}", f"Result: {result}"]
                except:
                    pass
        
        # If no specific pattern matched, provide a general mathematical analysis
        if solution == "Unable to determine solution":
            solution = f"Mathematical analysis of: {problem_text[:50]}..."
            confidence = 0.3
            reasoning = ["Problem requires further analysis", "Pattern not recognized in current knowledge base"]
        
        return {
            "task_id": task.get("task_id", "unknown"),
            "findings": solution,
            "confidence": confidence,
            "sources_analyzed": 1,
            "key_insights": reasoning,
            "processing_time": 1.5,
            "status": "completed",
            "problem_type": "mathematical"
        }
    
    def _general_research_task(self, task_description: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general research tasks"""
        result = {
            "task_id": task.get("task_id", "unknown"),
            "findings": f"Research findings for: {task_description}",
            "confidence": 0.85,
            "sources_analyzed": 5,
            "key_insights": [
                "Insight 1: Relevant finding from analysis",
                "Insight 2: Another important discovery"
            ],
            "processing_time": 2.5,
            "status": "completed"
        }
        
        return result


class ValidatorAgent(BaseAgent):
    """
    Validator agent specialized in reviewing and validating information
    """
    
    def __init__(self, agent_id: str, blackboard: IBlackboard):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.VALIDATOR,
            blackboard=blackboard,
            capabilities=["validation", "fact_checking", "quality_assessment", "peer_review"]
        )
        
        self.message_handlers[MessageType.KNOWLEDGE_SHARE] = self._handle_knowledge_validation
    
    def _handle_knowledge_validation(self, message: Message):
        """Validate shared knowledge"""
        if message.sender_role == AgentRole.RESEARCHER:
            findings = message.content.get("findings", {})
            validation_result = self.process_task({"validation_target": findings})
            
            # Share validation results
            self.share_knowledge(
                topic=f"validation_{message.content.get('topic')}",
                findings=validation_result,
                target_roles=[AgentRole.EDITOR, AgentRole.COORDINATOR]
            )
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a validation task"""
        logger.info(f"Validator {self.agent_id} validating content")
        
        # Mock validation process
        validation_result = {
            "validation_id": f"val_{int(time.time())}",
            "target": task.get("validation_target"),
            "accuracy_score": 0.92,
            "completeness_score": 0.88,
            "consistency_score": 0.95,
            "issues_found": ["Minor inconsistency in source citation"],
            "recommendations": ["Verify source #3", "Add supporting evidence"],
            "overall_quality": "high",
            "validated_by": self.agent_id,
            "status": "completed"
        }
        
        return validation_result



if __name__ == "__main__":
    print("Testing Base Agent Communication System")
    print("=" * 50)
    
    # Create blackboard with console output
    blackboard = create_blackboard()
    blackboard.enable_real_time_console()
    
    print("Creating and starting agents...")
    
    # Create agents
    researcher = ResearcherAgent("researcher_001", blackboard)
    validator = ValidatorAgent("validator_001", blackboard)
    
    # Start agents
    researcher.start()
    validator.start()
    
    print("\nSimulating collaborative task...")
    
    # Simulate a collaborative task
    researcher.request_task_assignment(
        "Research AI ethics considerations in multi-agent systems",
        ["research", "analysis"]
    )
    
    # Let the system run for a moment
    time.sleep(2)
    
    print("\nFinal Status Report:")
    print("-" * 30)
    
    # Show agent statuses
    researcher_status = researcher.get_status()
    validator_status = validator.get_status()
    
    print(f"Researcher: {researcher_status['status']} - Tasks: {researcher_status['completed_tasks']}")
    print(f"Validator: {validator_status['status']} - Tasks: {validator_status['completed_tasks']}")
    
    # Show blackboard statistics
    stats = blackboard.get_statistics()
    print(f"\nBlackboard: {stats['total_messages']} messages, {stats['memory_usage_mb']:.3f} MB")
    
    # Stop agents
    researcher.stop()
    validator.stop()
    
    blackboard.disable_real_time_console()
    print("\nTest completed successfully!")
