from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
import json
import uuid
import threading
import logging
from queue import Queue, Empty

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages that can be posted to the blackboard"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    KNOWLEDGE_SHARE = "knowledge_share"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"
    HUMAN_FEEDBACK = "human_feedback"
    ERROR_REPORT = "error_report"
    QUESTION = "question"
    RESPONSE = "response"
    NOTIFICATION = "notification"


class AgentRole(Enum):
    """Predefined agent roles as per the proposal"""
    RESEARCHER = "researcher"
    VALIDATOR = "validator"
    EDITOR = "editor"
    COORDINATOR = "coordinator"
    HUMAN_SUPERVISOR = "human_supervisor"


@dataclass
class Message:
    #Represents a message on the blackboard
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    sender_role: AgentRole = AgentRole.RESEARCHER
    message_type: MessageType = MessageType.COORDINATION
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1=low, 2=medium, 3=high
    requires_response: bool = False
    target_agents: List[str] = field(default_factory=list)  # Empty means broadcast
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'sender_role': self.sender_role.value,
            'message_type': self.message_type.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority,
            'requires_response': self.requires_response,
            'target_agents': self.target_agents,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            id=data['id'],
            sender_id=data['sender_id'],
            sender_role=AgentRole(data['sender_role']),
            message_type=MessageType(data['message_type']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            priority=data['priority'],
            requires_response=data['requires_response'],
            target_agents=data['target_agents'],
            metadata=data['metadata']
        )


@dataclass
class BlackboardEntry:
    """Represents an entry on the blackboard with access control"""
    message: Message
    access_level: str = "public"  # public, role_based, private
    allowed_roles: List[AgentRole] = field(default_factory=list)
    allowed_agents: List[str] = field(default_factory=list)
    expiry_time: Optional[datetime] = None
    read_count: int = 0
    
    def can_access(self, agent_id: str, agent_role: AgentRole) -> bool:
        #Check if agent can access this entry
        if self.access_level == "public":
            return True
        elif self.access_level == "role_based":
            return agent_role in self.allowed_roles
        elif self.access_level == "private":
            return agent_id in self.allowed_agents
        return False


class IBlackboard(ABC):
    """Abstract interface for blackboard communication"""
    
    @abstractmethod
    def post_message(self, message: Message, access_level: str = "public", 
                    allowed_roles: List[AgentRole] = None, 
                    allowed_agents: List[str] = None) -> bool:
        """Post a message to the blackboard"""
        pass
    
    @abstractmethod
    def read_messages(self, agent_id: str, agent_role: AgentRole, 
                     message_type: Optional[MessageType] = None,
                     since: Optional[datetime] = None) -> List[Message]:
        """Read messages from the blackboard"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get blackboard usage statistics"""
        pass


class Blackboard(IBlackboard):

    
    def __init__(self, max_messages: int = 10000):
        self.max_messages = max_messages
        self.entries: List[BlackboardEntry] = []
        self.lock = threading.RLock()
        self.real_time_console = False  # Enable/disable real-time console output
        self.stats = {
            'total_messages': 0,
            'messages_by_type': {},
            'messages_by_role': {}
        }
        
        logger.info(f"Blackboard initialized with max_messages={max_messages}")
    
    def post_message(self, message: Message, access_level: str = "public", 
                    allowed_roles: List[AgentRole] = None, 
                    allowed_agents: List[str] = None) -> bool:
        """Post a message to the blackboard with access control"""
        try:
            with self.lock:
                # Create blackboard entry
                entry = BlackboardEntry(
                    message=message,
                    access_level=access_level,
                    allowed_roles=allowed_roles or [],
                    allowed_agents=allowed_agents or []
                )
                
                # Add to blackboard
                self.entries.append(entry)
                
                # Maintain size limit
                if len(self.entries) > self.max_messages:
                    self.entries.pop(0)
                
                # Update statistics
                self.stats['total_messages'] += 1
                msg_type = message.message_type.value
                self.stats['messages_by_type'][msg_type] = self.stats['messages_by_type'].get(msg_type, 0) + 1
                role = message.sender_role.value
                self.stats['messages_by_role'][role] = self.stats['messages_by_role'].get(role, 0) + 1
                
                logger.info(f"Message posted: {message.id} from {message.sender_id} ({message.sender_role.value})")
                
                # Print message to console if enabled
                if self.real_time_console:
                    self._print_message_to_console(message)
                
                return True
                
        except Exception as e:
            logger.error(f"Error posting message: {str(e)}")
            return False
    
    def read_messages(self, agent_id: str, agent_role: AgentRole, 
                     message_type: Optional[MessageType] = None,
                     since: Optional[datetime] = None) -> List[Message]:
        """Read messages from the blackboard with filtering"""
        try:
            with self.lock:
                filtered_messages = []
                
                for entry in self.entries:
                    # Check access permissions
                    if not entry.can_access(agent_id, agent_role):
                        continue
                    
                    message = entry.message
                    
                    # Apply filters
                    if message_type and message.message_type != message_type:
                        continue
                    
                    if since and message.timestamp < since:
                        continue
                    
                    # Check if message is targeted
                    if message.target_agents and agent_id not in message.target_agents:
                        continue
                    
                    # Update read count
                    entry.read_count += 1
                    
                    filtered_messages.append(message)
                
                logger.info(f"Agent {agent_id} read {len(filtered_messages)} messages")
                return filtered_messages
                
        except Exception as e:
            logger.error(f"Error reading messages for {agent_id}: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get blackboard usage statistics"""
        with self.lock:
            return {
                **self.stats,
                'current_messages': len(self.entries),
                'max_messages': self.max_messages,
                'memory_usage_mb': self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        # Rough estimation based on average message size
        avg_message_size = 1024  # 1KB per message estimate
        return (len(self.entries) * avg_message_size) / (1024 * 1024)
    
    def clear_expired_messages(self):
        """Remove expired messages from the blackboard"""
        with self.lock:
            current_time = datetime.now()
            self.entries = [entry for entry in self.entries 
                          if entry.expiry_time is None or entry.expiry_time > current_time]
    
    def export_audit_trail(self) -> List[Dict[str, Any]]:
        """Export audit trail for analysis"""
        with self.lock:
            return [
                {
                    **entry.message.to_dict(),
                    'access_level': entry.access_level,
                    'read_count': entry.read_count,
                    'allowed_roles': [role.value for role in entry.allowed_roles],
                    'allowed_agents': entry.allowed_agents
                }
                for entry in self.entries
            ]
    
    def print_recent_messages(self, count: int = 10):
        """Print recent messages in human-readable format"""
        with self.lock:
            recent_entries = self.entries[-count:] if len(self.entries) > count else self.entries
            
            if not recent_entries:
                print("\nBLACKBOARD: No messages to display")
                return
            
            print(f"\nBLACKBOARD: Last {len(recent_entries)} Messages")
            print("=" * 60)
            
            for i, entry in enumerate(recent_entries, 1):
                msg = entry.message
                timestamp = msg.timestamp.strftime("%H:%M:%S")
                
                # Create readable message type
                msg_type_display = {
                    "task_request": "Task Request",
                    "task_response": "Task Response", 
                    "knowledge_share": "Knowledge Share",
                    "coordination": "Coordination",
                    "status_update": "Status Update",
                    "human_feedback": "Human Feedback",
                    "error_report": "Error Report"
                }.get(msg.message_type.value, msg.message_type.value.title())
                
                # Create readable role
                role_display = {
                    "researcher": "Researcher",
                    "validator": "Validator", 
                    "editor": "Editor",
                    "coordinator": "Coordinator",
                    "human_supervisor": "Human Supervisor"
                }.get(msg.sender_role.value, msg.sender_role.value.title())
                
                print(f"\n{i}. [{timestamp}] {msg_type_display}")
                print(f"   From: {role_display} ({msg.sender_id})")
                
                # Display content in readable format
                if msg.message_type == MessageType.KNOWLEDGE_SHARE:
                    topic = msg.content.get('topic', 'Unknown topic')
                    findings = msg.content.get('findings', {})
                    if isinstance(findings, dict):
                        confidence = findings.get('confidence', 'N/A')
                        print(f"   Topic: {topic}")
                        print(f"   Confidence: {confidence}")
                        if 'key_points' in findings:
                            print(f"   Key Points: {findings['key_points']}")
                    else:
                        print(f"   Topic: {topic}")
                        print(f"   Content: {findings}")
                
                elif msg.message_type == MessageType.TASK_REQUEST:
                    task_desc = msg.content.get('task_description', 'No description')
                    required_caps = msg.content.get('required_capabilities', [])
                    print(f"   Task: {task_desc}")
                    if required_caps:
                        print(f"   Required Skills: {', '.join(required_caps)}")
                
                elif msg.message_type == MessageType.STATUS_UPDATE:
                    status = msg.content.get('status', 'Unknown')
                    load = msg.content.get('load_factor', 0)
                    completed = msg.content.get('completed_tasks', 0)
                    print(f"   Status: {status}")
                    print(f"   Load: {load:.1%}, Tasks Completed: {completed}")
                
                elif msg.message_type == MessageType.ERROR_REPORT:
                    error_type = msg.content.get('error_type', 'Unknown error')
                    description = msg.content.get('description', 'No description')
                    print(f"   Error: {error_type}")
                    print(f"   Details: {description}")
                
                else:
                    # Generic content display
                    if msg.content:
                        for key, value in list(msg.content.items())[:3]:  # Show first 3 items
                            if isinstance(value, (str, int, float, bool)):
                                print(f"   {key.title()}: {value}")
                            elif isinstance(value, list) and len(value) <= 3:
                                print(f"   {key.title()}: {', '.join(map(str, value))}")
                
                # Show targets if specific
                if msg.target_agents:
                    print(f"   → To: {', '.join(msg.target_agents)}")
                
                # Show priority if high
                if msg.priority > 1:
                    priority_text = {2: "Medium", 3: "High"}.get(msg.priority, "Unknown")
                    print(f"   Priority: {priority_text}")
                
                print(f"   Access: {entry.access_level}, Read {entry.read_count} times")
            
            print("=" * 60)
    
    def _print_message_to_console(self, message: Message):
        """Print a single message to console in human-readable format"""
        print(f"\n--- NEW MESSAGE ---")
        print(f"Type: {message.message_type.value.upper()}")
        print(f"From: {message.sender_id} ({message.sender_role.value})")
        print(f"Time: {message.timestamp.strftime('%H:%M:%S')}")
        
        # Show content based on message type
        if message.message_type == MessageType.TASK_REQUEST:
            task_desc = message.content.get('task_description', 'No description')
            required_caps = message.content.get('required_capabilities', [])
            print(f"Task: {task_desc}")
            if required_caps:
                print(f"Required capabilities: {', '.join(required_caps)}")
                
        elif message.message_type == MessageType.KNOWLEDGE_SHARE:
            topic = message.content.get('topic', 'Unknown topic')
            findings = message.content.get('findings', {})
            print(f"Topic: {topic}")
            if isinstance(findings, dict):
                for key, value in findings.items():
                    print(f"  {key}: {value}")
            else:
                print(f"Content: {findings}")
                
        elif message.message_type == MessageType.STATUS_UPDATE:
            status = message.content.get('status', 'Unknown')
            completed = message.content.get('completed_tasks', 0)
            capabilities = message.content.get('capabilities', [])
            print(f"Status: {status}")
            print(f"Completed tasks: {completed}")
            print(f"Capabilities: {', '.join(capabilities)}")
            
        else:
            print(f"Content: {message.content}")
        
        if message.priority > 1:
            priority_text = {2: "Medium", 3: "High"}.get(message.priority, "Unknown")
            print(f"Priority: {priority_text}")
        
        if message.target_agents:
            print(f"Targeted to: {', '.join(message.target_agents)}")
            
        print("-" * 40)
    
    def enable_real_time_console(self):
        """Enable real-time console output for messages"""
        self.real_time_console = True
        print("Real-time console output ENABLED")
    
    def disable_real_time_console(self):
        """Disable real-time console output for messages"""
        self.real_time_console = False
        print("Real-time console output DISABLED")


# Factory function for creating blackboards
def create_blackboard(blackboard_type: str = "standard", **kwargs) -> IBlackboard:
    """Factory function to create different types of blackboards"""
    if blackboard_type == "standard":
        return Blackboard(**kwargs)
    else:
        raise ValueError(f"Unknown blackboard type: {blackboard_type}")


if __name__ == "__main__":
    # Basic test of the blackboard system
    blackboard = create_blackboard()
    
    # Create test message
    test_message = Message(
        sender_id="agent_1",
        sender_role=AgentRole.RESEARCHER,
        message_type=MessageType.KNOWLEDGE_SHARE,
        content={"topic": "AI Ethics", "findings": "Initial research findings on AI ethics considerations"},
        requires_response=True
    )
    
    # Post message
    success = blackboard.post_message(test_message)
    print(f"Message posted: {success}")
    
    # Read messages
    messages = blackboard.read_messages("agent_2", AgentRole.VALIDATOR)
    print(f"Messages read: {len(messages)}")
    
    # Show statistics
    stats = blackboard.get_statistics()
    print(f"Blackboard stats: {stats}")
