"""
CollabArena Dataset Integration Module

This module handles integration with the Hugging Face dataset (Can111/m500)
and provides data processing capabilities for multi-agent collaboration tasks.
"""

from typing import Dict, List, Optional, Iterator, Any
import logging
from datetime import datetime
import json
import os
from dataclasses import dataclass

try:
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("Hugging Face libraries not available. Install with: pip install datasets transformers")

from blackboard_communication import Message, MessageType, AgentRole

logger = logging.getLogger(__name__)


@dataclass
class DatasetSample:
    """Represents a sample from the dataset"""
    id: str
    content: str
    metadata: Dict[str, Any]
    source: str = "m500"
    processed_by: List[str] = None
    
    def __post_init__(self):
        if self.processed_by is None:
            self.processed_by = []


class DatasetManager:
    """
    Manages dataset loading, processing, and distribution for multi-agent tasks
    """
    
    def __init__(self, dataset_name: str = "Can111/m500", cache_dir: Optional[str] = None):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.dataset = None
        self.samples = []
        self.current_index = 0
        self.tokenizer = None
        
        if not HF_AVAILABLE:
            logger.error("Hugging Face libraries required for dataset integration")
            return
            
        self._load_dataset()
        self._initialize_tokenizer()
    
    def _load_dataset(self):
        """Load the dataset from Hugging Face"""
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            self.dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)
            
            # Process dataset into our format
            self._process_dataset()
            
            logger.info(f"Dataset loaded successfully. Total samples: {len(self.samples)}")
            
        except Exception as e:
            logger.error(f"Error loading dataset {self.dataset_name}: {str(e)}")
            # Create mock data for testing when dataset is not available
            self._create_mock_data()
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for text processing"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            logger.info("Tokenizer initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize tokenizer: {str(e)}")
    
    def _process_dataset(self):
        """Process the raw dataset into DatasetSample objects"""
        if not self.dataset:
            return
        
        try:
            # Assuming the dataset has 'text' and potentially other fields
            # Adjust based on actual dataset structure
            for split_name, split_data in self.dataset.items():
                for i, item in enumerate(split_data):
                    sample = DatasetSample(
                        id=f"{split_name}_{i}",
                        content=self._extract_content(item),
                        metadata={
                            "split": split_name,
                            "original_index": i,
                            "length": len(str(item)),
                            **self._extract_metadata(item)
                        }
                    )
                    self.samples.append(sample)
                    
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            self._create_mock_data()
    
    def _extract_content(self, item: Dict[str, Any]) -> str:
        """Extract main content from a dataset item"""
        # Try common field names for content
        content_fields = ['text', 'content', 'input', 'question', 'prompt']
        
        for field in content_fields:
            if field in item:
                return str(item[field])
        
        # If no standard field found, convert entire item to string
        return str(item)
    
    def _extract_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from a dataset item"""
        # Extract useful metadata while excluding the main content
        metadata = {}
        content_fields = ['text', 'content', 'input']
        
        for key, value in item.items():
            if key not in content_fields:
                metadata[key] = value
        
        return metadata
    
    def _create_mock_data(self):
        """Create mock data for testing when real dataset is unavailable"""
        mock_samples = [
            {
                "content": "Research the impact of artificial intelligence on workplace collaboration and productivity.",
                "metadata": {"topic": "AI workplace impact", "complexity": "medium"}
            },
            {
                "content": "Analyze the ethical considerations in multi-agent AI systems, focusing on decision-making transparency.",
                "metadata": {"topic": "AI ethics", "complexity": "high"}
            },
            {
                "content": "Investigate communication protocols in distributed AI systems and their effectiveness.",
                "metadata": {"topic": "AI communication", "complexity": "medium"}
            },
            {
                "content": "Examine the role of human oversight in automated multi-agent collaborative processes.",
                "metadata": {"topic": "human-AI collaboration", "complexity": "high"}
            },
            {
                "content": "Study memory management strategies in large-scale multi-agent systems.",
                "metadata": {"topic": "memory management", "complexity": "medium"}
            }
        ]
        
        for i, mock_item in enumerate(mock_samples):
            sample = DatasetSample(
                id=f"mock_{i}",
                content=mock_item["content"],
                metadata={
                    **mock_item["metadata"],
                    "source": "mock_data",
                    "created_at": datetime.now().isoformat()
                }
            )
            self.samples.append(sample)
        
        logger.info(f"Created {len(mock_samples)} mock samples for testing")
    
    def get_sample(self, sample_id: str) -> Optional[DatasetSample]:
        """Get a specific sample by ID"""
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        return None
    
    def get_samples_by_topic(self, topic: str) -> List[DatasetSample]:
        """Get samples related to a specific topic"""
        matching_samples = []
        topic_lower = topic.lower()
        
        for sample in self.samples:
            # Check content and metadata for topic relevance
            if (topic_lower in sample.content.lower() or 
                topic_lower in str(sample.metadata).lower()):
                matching_samples.append(sample)
        
        return matching_samples
    
    def get_next_batch(self, batch_size: int = 5) -> List[DatasetSample]:
        """Get the next batch of samples"""
        if self.current_index >= len(self.samples):
            self.current_index = 0  # Reset to beginning
        
        end_index = min(self.current_index + batch_size, len(self.samples))
        batch = self.samples[self.current_index:end_index]
        self.current_index = end_index
        
        return batch
    
    def create_collaborative_task(self, sample: DatasetSample, 
                                task_type: str = "research_analysis") -> Dict[str, Any]:
        """Create a collaborative task from a dataset sample"""
        task = {
            "task_id": f"task_{sample.id}_{int(datetime.now().timestamp())}",
            "task_type": task_type,
            "source_sample_id": sample.id,
            "description": sample.content,
            "metadata": sample.metadata,
            "required_roles": [AgentRole.RESEARCHER, AgentRole.VALIDATOR],
            "estimated_complexity": sample.metadata.get("complexity", "medium"),
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        return task
    
    def get_random_sample(self) -> Optional[DatasetSample]:
        """Get a random sample from the dataset"""
        if not self.samples:
            return None
        import random
        return random.choice(self.samples)
    
    def get_sample_count(self) -> int:
        """Get the total number of samples"""
        return len(self.samples)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.samples:
            return {"error": "No samples available"}
        
        # Calculate statistics
        total_samples = len(self.samples)
        avg_content_length = sum(len(sample.content) for sample in self.samples) / total_samples
        
        # Topic distribution
        topics = {}
        for sample in self.samples:
            topic = sample.metadata.get("topic", "unknown")
            topics[topic] = topics.get(topic, 0) + 1
        
        return {
            "total_samples": total_samples,
            "average_content_length": round(avg_content_length, 2),
            "topic_distribution": topics,
            "current_index": self.current_index,
            "dataset_name": self.dataset_name,
            "has_tokenizer": self.tokenizer is not None
        }


class TaskGenerator:
    """
    Generates collaborative tasks from dataset samples for multi-agent systems
    """
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.task_templates = {
            "research_synthesis": {
                "description": "Research and synthesize information on: {content}",
                "roles": [AgentRole.RESEARCHER, AgentRole.VALIDATOR, AgentRole.EDITOR],
                "steps": ["research", "validate", "synthesize", "review"]
            },
            "conflict_resolution": {
                "description": "Analyze conflicting information and provide resolution for: {content}",
                "roles": [AgentRole.RESEARCHER, AgentRole.VALIDATOR],
                "steps": ["analyze_conflicts", "validate_sources", "propose_resolution"]
            },
            "collaborative_analysis": {
                "description": "Perform collaborative analysis on: {content}",
                "roles": [AgentRole.RESEARCHER, AgentRole.VALIDATOR, AgentRole.EDITOR],
                "steps": ["initial_analysis", "peer_review", "consensus_building"]
            }
        }
    
    def generate_task_from_sample(self, sample: DatasetSample, 
                                task_type: str = "research_synthesis") -> Dict[str, Any]:
        """Generate a specific type of collaborative task from a sample"""
        if task_type not in self.task_templates:
            task_type = "research_synthesis"
        
        template = self.task_templates[task_type]
        
        task = {
            "task_id": f"{task_type}_{sample.id}_{int(datetime.now().timestamp())}",
            "task_type": task_type,
            "source_sample": sample,
            "description": template["description"].format(content=sample.content),
            "required_roles": template["roles"],
            "workflow_steps": template["steps"],
            "metadata": {
                **sample.metadata,
                "task_template": task_type,
                "estimated_duration": self._estimate_task_duration(sample, task_type)
            },
            "created_at": datetime.now().isoformat(),
            "status": "ready"
        }
        
        return task
    
    def _estimate_task_duration(self, sample: DatasetSample, task_type: str) -> int:
        """Estimate task duration in minutes based on content and type"""
        base_duration = {
            "research_synthesis": 15,
            "conflict_resolution": 20,
            "collaborative_analysis": 25
        }
        
        content_length_factor = len(sample.content) / 1000  # Per 1000 characters
        complexity_factor = {"low": 0.8, "medium": 1.0, "high": 1.5}.get(
            sample.metadata.get("complexity", "medium"), 1.0
        )
        
        estimated_duration = int(
            base_duration.get(task_type, 15) * (1 + content_length_factor) * complexity_factor
        )
        
        return max(5, estimated_duration)  # Minimum 5 minutes
    
    def generate_batch_tasks(self, batch_size: int = 3, 
                           task_types: List[str] = None) -> List[Dict[str, Any]]:
        """Generate a batch of tasks for multi-agent collaboration"""
        if task_types is None:
            task_types = list(self.task_templates.keys())
        
        samples = self.dataset_manager.get_next_batch(batch_size)
        tasks = []
        
        for i, sample in enumerate(samples):
            task_type = task_types[i % len(task_types)]
            task = self.generate_task_from_sample(sample, task_type)
            tasks.append(task)
        
        return tasks


# Example usage and testing
if __name__ == "__main__":
    # Initialize dataset manager
    dataset_manager = DatasetManager()
    
    # Show dataset statistics
    print("Dataset Statistics:")
    stats = dataset_manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create task generator
    task_generator = TaskGenerator(dataset_manager)
    
    # Generate some sample tasks
    print("\nGenerated Tasks:")
    tasks = task_generator.generate_batch_tasks(batch_size=3)
    
    for i, task in enumerate(tasks, 1):
        print(f"\nTask {i}:")
        print(f"  ID: {task['task_id']}")
        print(f"  Type: {task['task_type']}")
        print(f"  Description: {task['description'][:100]}...")
        print(f"  Required Roles: {[role.value for role in task['required_roles']]}")
        print(f"  Estimated Duration: {task['metadata']['estimated_duration']} minutes")
    
    # Test topic-based sample retrieval
    print(f"\nSamples related to 'AI':")
    ai_samples = dataset_manager.get_samples_by_topic("AI")
    for sample in ai_samples[:2]:  # Show first 2
        print(f"  - {sample.id}: {sample.content[:80]}...")
