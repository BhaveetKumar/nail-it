# 🎯 Prompt Engineering

> **Master Prompt Engineering: from basic techniques to advanced strategies for LLMs**

## 🎯 **Learning Objectives**

- Understand prompt engineering fundamentals and best practices
- Master different prompt engineering techniques and patterns
- Implement prompt optimization and evaluation strategies
- Handle few-shot learning and chain-of-thought prompting
- Build production-ready prompt engineering systems

## 📚 **Table of Contents**

1. [Prompt Engineering Fundamentals](#prompt-engineering-fundamentals)
2. [Advanced Techniques](#advanced-techniques)
3. [Few-Shot Learning](#few-shot-learning)
4. [Chain-of-Thought Prompting](#chain-of-thought-prompting)
5. [Production Implementation](#production-implementation)
6. [Interview Questions](#interview-questions)

---

## 🎯 **Prompt Engineering Fundamentals**

### **Core Concepts**

#### **Concept**
Prompt engineering is the practice of designing and optimizing input prompts to get desired outputs from large language models.

#### **Key Principles**
- **Clarity**: Clear and specific instructions
- **Context**: Provide relevant context and examples
- **Format**: Specify desired output format
- **Constraints**: Set appropriate boundaries and limitations
- **Iteration**: Continuously refine and improve prompts

#### **Code Example**

```python
import openai
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptType(Enum):
    """Types of prompts"""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ROLE_BASED = "role_based"
    TEMPLATE = "template"

@dataclass
class PromptConfig:
    """Prompt configuration"""
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None

class PromptEngineer:
    """Prompt engineering system"""
    
    def __init__(self, api_key: str, config: PromptConfig):
        self.api_key = api_key
        self.config = config
        self.prompt_templates = {}
        self.prompt_history = []
        
        # Initialize OpenAI client
        openai.api_key = api_key
    
    def create_zero_shot_prompt(self, task: str, input_data: str, output_format: str = "text") -> str:
        """Create zero-shot prompt"""
        prompt = f"""
Task: {task}

Input: {input_data}

Please provide your response in {output_format} format.
"""
        return prompt.strip()
    
    def create_few_shot_prompt(self, task: str, examples: List[Dict[str, str]], input_data: str) -> str:
        """Create few-shot prompt with examples"""
        prompt = f"Task: {task}\n\n"
        
        # Add examples
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
        
        # Add current input
        prompt += f"Input: {input_data}\n"
        prompt += "Output:"
        
        return prompt.strip()
    
    def create_chain_of_thought_prompt(self, problem: str, reasoning_steps: bool = True) -> str:
        """Create chain-of-thought prompt"""
        prompt = f"""
Problem: {problem}

Please solve this step by step:
"""
        
        if reasoning_steps:
            prompt += """
1. First, understand what is being asked
2. Break down the problem into smaller parts
3. Solve each part systematically
4. Combine the results
5. Verify your answer

Let's work through this step by step:
"""
        
        return prompt.strip()
    
    def create_role_based_prompt(self, role: str, task: str, context: str, input_data: str) -> str:
        """Create role-based prompt"""
        prompt = f"""
You are a {role}.

Context: {context}

Task: {task}

Input: {input_data}

Please respond as a {role} would:
"""
        return prompt.strip()
    
    def create_template_prompt(self, template_name: str, variables: Dict[str, str]) -> str:
        """Create prompt from template"""
        if template_name not in self.prompt_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.prompt_templates[template_name]
        
        # Replace variables in template
        prompt = template
        for key, value in variables.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        
        return prompt
    
    def add_template(self, name: str, template: str):
        """Add a new prompt template"""
        self.prompt_templates[name] = template
        logger.info(f"Added template: {name}")
    
    def execute_prompt(self, prompt: str, config: Optional[PromptConfig] = None) -> Dict[str, Any]:
        """Execute a prompt and get response"""
        if config is None:
            config = self.config
        
        try:
            response = openai.ChatCompletion.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop
            )
            
            result = {
                "prompt": prompt,
                "response": response.choices[0].message.content,
                "usage": response.usage,
                "model": response.model,
                "timestamp": time.time()
            }
            
            # Store in history
            self.prompt_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing prompt: {e}")
            return {
                "prompt": prompt,
                "response": None,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def batch_execute_prompts(self, prompts: List[str], config: Optional[PromptConfig] = None) -> List[Dict[str, Any]]:
        """Execute multiple prompts in batch"""
        results = []
        
        for prompt in prompts:
            result = self.execute_prompt(prompt, config)
            results.append(result)
            
            # Add delay to avoid rate limiting
            time.sleep(0.1)
        
        return results
    
    def evaluate_prompt(self, prompt: str, expected_output: str, config: Optional[PromptConfig] = None) -> Dict[str, Any]:
        """Evaluate prompt performance"""
        result = self.execute_prompt(prompt, config)
        
        if result["response"] is None:
            return {
                "prompt": prompt,
                "success": False,
                "error": result.get("error", "Unknown error"),
                "score": 0.0
            }
        
        # Simple evaluation metrics
        response = result["response"].lower()
        expected = expected_output.lower()
        
        # Calculate similarity score
        similarity_score = self._calculate_similarity(response, expected)
        
        # Check if key concepts are present
        key_concepts = self._extract_key_concepts(expected_output)
        concept_score = self._calculate_concept_score(response, key_concepts)
        
        # Overall score
        overall_score = (similarity_score + concept_score) / 2
        
        return {
            "prompt": prompt,
            "response": result["response"],
            "expected": expected_output,
            "similarity_score": similarity_score,
            "concept_score": concept_score,
            "overall_score": overall_score,
            "success": overall_score > 0.7,
            "usage": result.get("usage", {})
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        # Simple word overlap similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple keyword extraction
        words = text.split()
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        key_concepts = [word for word in words if word not in stop_words and len(word) > 3]
        return key_concepts
    
    def _calculate_concept_score(self, response: str, key_concepts: List[str]) -> float:
        """Calculate concept coverage score"""
        if not key_concepts:
            return 1.0
        
        response_lower = response.lower()
        covered_concepts = sum(1 for concept in key_concepts if concept.lower() in response_lower)
        
        return covered_concepts / len(key_concepts)
    
    def optimize_prompt(self, base_prompt: str, examples: List[Dict[str, str]], iterations: int = 5) -> str:
        """Optimize prompt through iterative improvement"""
        best_prompt = base_prompt
        best_score = 0.0
        
        for iteration in range(iterations):
            # Evaluate current prompt
            scores = []
            for example in examples:
                result = self.evaluate_prompt(best_prompt, example["expected"])
                scores.append(result["overall_score"])
            
            avg_score = sum(scores) / len(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_prompt = base_prompt
            
            # Generate variations
            variations = self._generate_prompt_variations(best_prompt)
            
            # Test variations
            for variation in variations:
                var_scores = []
                for example in examples:
                    result = self.evaluate_prompt(variation, example["expected"])
                    var_scores.append(result["overall_score"])
                
                var_avg_score = sum(var_scores) / len(var_scores)
                
                if var_avg_score > best_score:
                    best_score = var_avg_score
                    best_prompt = variation
            
            logger.info(f"Iteration {iteration + 1}: Best score = {best_score:.3f}")
        
        return best_prompt
    
    def _generate_prompt_variations(self, prompt: str) -> List[str]:
        """Generate variations of a prompt"""
        variations = []
        
        # Add clarity variations
        variations.append(f"Please be very clear and specific in your response.\n\n{prompt}")
        variations.append(f"Think step by step and provide a detailed response.\n\n{prompt}")
        
        # Add format variations
        variations.append(f"{prompt}\n\nPlease format your response clearly.")
        variations.append(f"{prompt}\n\nProvide your answer in a structured format.")
        
        # Add context variations
        variations.append(f"Given the following context, {prompt.lower()}")
        variations.append(f"Consider all relevant information when responding: {prompt}")
        
        return variations
    
    def get_prompt_history(self) -> List[Dict[str, Any]]:
        """Get prompt execution history"""
        return self.prompt_history
    
    def analyze_prompt_performance(self) -> Dict[str, Any]:
        """Analyze prompt performance metrics"""
        if not self.prompt_history:
            return {"error": "No prompt history available"}
        
        successful_prompts = [p for p in self.prompt_history if p.get("response") is not None]
        failed_prompts = [p for p in self.prompt_history if p.get("response") is None]
        
        total_tokens = sum(p.get("usage", {}).get("total_tokens", 0) for p in successful_prompts)
        avg_tokens = total_tokens / len(successful_prompts) if successful_prompts else 0
        
        return {
            "total_prompts": len(self.prompt_history),
            "successful_prompts": len(successful_prompts),
            "failed_prompts": len(failed_prompts),
            "success_rate": len(successful_prompts) / len(self.prompt_history),
            "total_tokens": total_tokens,
            "average_tokens_per_prompt": avg_tokens,
            "templates_used": len(self.prompt_templates)
        }

# Example usage
def main():
    # Create prompt configuration
    config = PromptConfig(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
        top_p=1.0
    )
    
    # Initialize prompt engineer
    # Note: You'll need to set your OpenAI API key
    prompt_engineer = PromptEngineer("your-api-key-here", config)
    
    # Add some templates
    prompt_engineer.add_template(
        "code_review",
        """
You are an expert code reviewer. Review the following code:

Code:
{code}

Please provide:
1. Code quality assessment
2. Potential issues or bugs
3. Suggestions for improvement
4. Security considerations
"""
    )
    
    prompt_engineer.add_template(
        "explanation",
        """
Explain the following concept in simple terms:

Concept: {concept}
Audience: {audience}
Level: {level}

Please provide a clear, concise explanation with examples.
"""
    )
    
    # Example 1: Zero-shot prompt
    zero_shot_prompt = prompt_engineer.create_zero_shot_prompt(
        "Summarize the following text",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "bullet points"
    )
    
    print("Zero-shot prompt:")
    print(zero_shot_prompt)
    print()
    
    # Example 2: Few-shot prompt
    examples = [
        {
            "input": "The weather is sunny today.",
            "output": "Positive"
        },
        {
            "input": "I failed my exam.",
            "output": "Negative"
        }
    ]
    
    few_shot_prompt = prompt_engineer.create_few_shot_prompt(
        "Classify the sentiment of the following text",
        examples,
        "I love this new product!"
    )
    
    print("Few-shot prompt:")
    print(few_shot_prompt)
    print()
    
    # Example 3: Chain-of-thought prompt
    cot_prompt = prompt_engineer.create_chain_of_thought_prompt(
        "If a train travels 120 miles in 2 hours, what is its average speed?"
    )
    
    print("Chain-of-thought prompt:")
    print(cot_prompt)
    print()
    
    # Example 4: Role-based prompt
    role_prompt = prompt_engineer.create_role_based_prompt(
        "data scientist",
        "Analyze this dataset and provide insights",
        "You are working with customer data for an e-commerce company",
        "Customer purchase history with 1000 records"
    )
    
    print("Role-based prompt:")
    print(role_prompt)
    print()
    
    # Example 5: Template prompt
    template_prompt = prompt_engineer.create_template_prompt(
        "code_review",
        {
            "code": "def add(a, b):\n    return a + b"
        }
    )
    
    print("Template prompt:")
    print(template_prompt)
    print()
    
    # Example 6: Prompt optimization
    optimization_examples = [
        {
            "input": "What is machine learning?",
            "expected": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
        },
        {
            "input": "How does deep learning work?",
            "expected": "Deep learning uses neural networks with multiple layers to process data and learn complex patterns, similar to how the human brain processes information."
        }
    ]
    
    base_prompt = "Answer the following question: {question}"
    
    # Note: This would require actual API calls to work
    # optimized_prompt = prompt_engineer.optimize_prompt(base_prompt, optimization_examples)
    # print(f"Optimized prompt: {optimized_prompt}")
    
    # Analyze performance
    performance = prompt_engineer.analyze_prompt_performance()
    print("Performance analysis:")
    print(json.dumps(performance, indent=2))

if __name__ == "__main__":
    main()
```

---

## 🎯 **Interview Questions**

### **Prompt Engineering Theory**

#### **Q1: What is prompt engineering and why is it important?**
**Answer**: 
- **Definition**: The practice of designing and optimizing input prompts to get desired outputs from LLMs
- **Importance**: 
  - Improves model performance and accuracy
  - Reduces need for fine-tuning
  - Enables better control over model behavior
  - Makes models more accessible to non-technical users
  - Reduces costs and computational requirements

#### **Q2: What are the key principles of effective prompt engineering?**
**Answer**: 
- **Clarity**: Clear and specific instructions
- **Context**: Provide relevant context and background
- **Examples**: Use few-shot examples when possible
- **Format**: Specify desired output format
- **Constraints**: Set appropriate boundaries and limitations
- **Iteration**: Continuously refine and improve prompts

#### **Q3: What are the different types of prompting techniques?**
**Answer**: 
- **Zero-shot**: Direct question without examples
- **Few-shot**: Provide examples before the actual task
- **Chain-of-thought**: Ask model to show reasoning steps
- **Role-based**: Assign specific roles to the model
- **Template-based**: Use structured templates
- **Instruction tuning**: Fine-tune on instruction-following data

#### **Q4: How do you evaluate prompt effectiveness?**
**Answer**: 
- **Accuracy**: Measure correctness of outputs
- **Consistency**: Check for consistent responses
- **Relevance**: Ensure outputs match the task
- **Completeness**: Verify all required information is provided
- **Efficiency**: Measure token usage and response time
- **User satisfaction**: Collect feedback from end users

#### **Q5: What are common prompt engineering challenges?**
**Answer**: 
- **Ambiguity**: Unclear or ambiguous instructions
- **Context length**: Limited context window
- **Bias**: Model biases affecting outputs
- **Inconsistency**: Varying quality of responses
- **Cost**: High token usage for complex prompts
- **Maintenance**: Keeping prompts updated and relevant

### **Implementation Questions**

#### **Q6: Implement a prompt engineering system**
**Answer**: See the implementation above with different prompt types, evaluation, and optimization.

#### **Q7: How would you optimize prompts for production use?**
**Answer**: 
- **A/B Testing**: Test different prompt variations
- **Performance Monitoring**: Track prompt effectiveness
- **Automated Evaluation**: Use metrics to assess quality
- **User Feedback**: Incorporate user feedback
- **Cost Optimization**: Balance quality with token usage
- **Version Control**: Track prompt changes and versions

#### **Q8: How do you handle prompt injection attacks?**
**Answer**: 
- **Input Validation**: Validate and sanitize inputs
- **Prompt Isolation**: Separate user input from system prompts
- **Output Filtering**: Filter potentially harmful outputs
- **Rate Limiting**: Limit prompt execution frequency
- **Monitoring**: Monitor for suspicious patterns
- **User Education**: Educate users about safe practices

---

## 🚀 **Next Steps**

1. **Practice**: Implement all prompt engineering techniques
2. **Optimize**: Focus on prompt effectiveness and efficiency
3. **Deploy**: Build production prompt engineering systems
4. **Extend**: Learn about advanced techniques and best practices
5. **Interview**: Practice prompt engineering interview questions

---

**Ready to learn about Fine-tuning LLMs? Let's move to the next section!** 🎯
