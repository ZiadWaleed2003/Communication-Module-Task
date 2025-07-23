# CollabArena Multi-Agent Communication System

A comprehensive multi-agent collaboration system implementing the blackboard communication pattern for LLM-based agents, as described in the CollabArena research proposal.

## Overview

This system demonstrates multi-agent collaboration using a blackboard communication architecture where agents can:
- Share knowledge through a centralized blackboard
- Coordinate tasks and workflows
- Validate and review each other's work
- Maintain audit trails for research analysis

## System Architecture

### Core Components

1. **Blackboard Communication System** (`blackboard_communication.py`)
   - Centralized message board for agent communication
   - Role-based access control (RBAC)
   - Real-time message subscriptions
   - Audit trail logging

2. **Base Agent Framework** (`base_agent.py`)
   - Abstract base class for all agents
   - Built-in blackboard integration
   - Memory management
   - Status reporting

3. **Specialized Agents**
   - **ResearcherAgent**: Gathers and analyzes information
   - **ValidatorAgent**: Reviews and validates findings
   - **CoordinatorAgent**: Manages workflows and task distribution

4. **Dataset Integration** (`dataset_integration.py`)
   - Integrates with Hugging Face datasets (specifically Can111/m500)
   - Generates collaborative tasks from dataset samples
   - Falls back to mock data if dataset unavailable

5. **Main System** (`main.py`)
   - Orchestrates the entire multi-agent system
   - Provides demonstration workflows
   - Generates research reports and audit trails

## Features

### Communication Architecture
- **Blackboard Pattern**: Centralized, persistent communication
- **Message Types**: Task requests, knowledge sharing, coordination, status updates
- **Access Control**: Public, role-based, and private message access
- **Real-time Notifications**: Agents can subscribe to relevant message types

### Agent Collaboration
- **Task Distribution**: Automated task assignment based on agent roles and capabilities
- **Knowledge Sharing**: Agents share findings with relevant team members
- **Conflict Resolution**: Validation agents review and resolve inconsistencies
- **Human Oversight**: Integration points for human feedback and approval

### Research Features
- **Performance Metrics**: Success rates, completion times, collaboration efficiency
- **Audit Trails**: Complete message history for analysis
- **Agent Analytics**: Individual agent performance tracking
- **Collaboration Patterns**: Analysis of inter-agent communication

## Installation

1. **Clone or download the project files to your workspace**

2. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

3. **Optional: Install Hugging Face libraries for dataset integration:**
```bash
pip install datasets transformers
```

## Quick Start

### Basic Demo
Run the complete system demonstration:

```bash
python main.py
```

This will:
- Initialize the blackboard communication system
- Create researcher and validator agents
- Run collaborative tasks using the dataset
- Generate performance reports and audit trails

### Individual Component Testing

#### Test Blackboard Communication:
```bash
python blackboard_communication.py
```

#### Test Agent Interactions:
```bash
python base_agent.py
```

#### Test Dataset Integration:
```bash
python dataset_integration.py
```

#### Test Coordinator:
```bash
python coordinator_agent.py
```

## Usage Examples

### Creating a Custom Agent
```python
from base_agent import BaseAgent
from blackboard_communication import AgentRole, MessageType

class CustomAgent(BaseAgent):
    def __init__(self, agent_id: str, blackboard):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.RESEARCHER,
            blackboard=blackboard,
            capabilities=["custom_analysis", "specialized_task"]
        )
    
    def process_task(self, task):
        # Implement custom task processing
        return {"status": "completed", "result": "Custom analysis complete"}
```

### Setting Up a Collaboration Session
```python
from main import CollabArenaSystem

# Configure system
config = {
    "agents": {"researchers": 3, "validators": 2},
    "max_blackboard_messages": 10000
}

# Initialize and run
system = CollabArenaSystem(config)
system.initialize_system()
system.start_system()

# Run collaboration
report = system.run_collaboration_demo(num_tasks=5, duration_minutes=10)
```

## Dataset Integration

The system is designed to work with the Hugging Face dataset `Can111/m500`. If this dataset is not available, the system automatically falls back to mock data for testing.

### Dataset Features:
- Automatic task generation from dataset samples
- Topic-based sample filtering
- Collaborative task templates (research synthesis, conflict resolution)
- Metadata extraction for task complexity estimation

## Configuration Options

### System Configuration
```python
config = {
    "max_blackboard_messages": 10000,  # Maximum messages in blackboard
    "dataset_name": "Can111/m500",     # Hugging Face dataset
    "cache_dir": "./cache",            # Dataset cache directory
    "agents": {
        "researchers": 2,              # Number of researcher agents
        "validators": 1,               # Number of validator agents
        "editors": 1                   # Number of editor agents
    }
}
```

### Agent Configuration
- **Capabilities**: Define what tasks each agent can handle
- **Memory Management**: Local vs shared memory options
- **Communication Preferences**: Message type subscriptions
- **Performance Limits**: Maximum concurrent tasks per agent

## Research Applications

This system is designed to support research in:

### RQ1: Communication & Memory Architecture
- Compare blackboard vs direct messaging vs pub/sub patterns
- Analyze shared vs isolated vs RBAC memory configurations
- Measure impact on task success rates and efficiency

### RQ2: Planning & Delegation
- Study different task distribution strategies
- Analyze coordination overhead and solution coherence
- Compare capability-based vs planner-based delegation

### RQ3: Human Interaction
- Evaluate trust metrics with different transparency levels
- Measure cognitive load with various interaction models
- Study approval workflows and feedback integration

## Output and Analysis

### Generated Reports
- **Collaboration Reports**: Task completion rates, timing analysis
- **Agent Performance**: Individual agent metrics and error rates
- **Communication Analysis**: Message patterns and efficiency metrics
- **Research Insights**: Findings relevant to CollabArena research questions

### Audit Trails
Complete system logs including:
- All inter-agent messages
- Task assignment and completion records
- Agent status changes and error reports
- Performance metrics over time

### File Outputs
- `collabarena.log`: System activity log
- `collabarena_audit_YYYYMMDD_HHMMSS.json`: Complete audit trail
- Console output with real-time status updates

## Extending the System

### Adding New Agent Types
1. Inherit from `BaseAgent`
2. Implement `process_task()` method
3. Define role-specific capabilities
4. Set up message handlers for specialized communication

### Adding Communication Patterns
1. Extend the `IBlackboard` interface
2. Implement new message routing logic
3. Add access control mechanisms
4. Update audit trail recording

### Adding Task Types
1. Create new task templates in `TaskGenerator`
2. Define workflow steps and required roles
3. Implement task-specific validation logic
4. Add performance metrics tracking

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Dataset Loading**: System will use mock data if dataset unavailable
3. **Agent Communication**: Check blackboard initialization and agent subscriptions
4. **Performance**: Adjust `max_blackboard_messages` for large-scale testing

### Debug Mode
Enable detailed logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Research Citation

If you use this system in your research, please cite the CollabArena project:

```
CollabArena: Evaluating Memory and Communication Architectures 
for Multi-Agent LLM Collaboration
[Research Proposal, 2025]
```

## License

This code is provided for research and educational purposes. Please refer to the CollabArena project documentation for specific licensing terms.

## Contact

For questions about the system implementation or research applications, please refer to the CollabArena project documentation or contact the research team.

---

**Note**: This implementation represents the beginning of the CollabArena communication system as described in the research proposal. It provides a solid foundation for experimentation with different communication architectures, memory configurations, and multi-agent collaboration patterns.
