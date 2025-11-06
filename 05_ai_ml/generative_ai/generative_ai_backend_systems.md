---
# Auto-generated front matter
Title: Generative Ai Backend Systems
LastUpdated: 2025-11-06T20:45:58.322183
Tags: []
Status: draft
---

# Generative AI Backend Systems

Comprehensive guide to building backend systems for generative AI applications.

## 🎯 Generative AI Fundamentals

### Large Language Models (LLMs) Integration
```python
# LLM Service Architecture
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
import json
from dataclasses import dataclass
from enum import Enum

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"

@dataclass
class LLMRequest:
    prompt: str
    model: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = None
    stream: bool = False

@dataclass
class LLMResponse:
    text: str
    usage: Dict[str, int]
    model: str
    finish_reason: str
    metadata: Dict[str, Any] = None

class LLMService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {
            ModelProvider.OPENAI: self._call_openai,
            ModelProvider.ANTHROPIC: self._call_anthropic,
            ModelProvider.GOOGLE: self._call_google,
            ModelProvider.COHERE: self._call_cohere
        }
    
    async def generate(self, request: LLMRequest, provider: ModelProvider) -> LLMResponse:
        """Generate text using specified LLM provider"""
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return await self.providers[provider](request)
    
    async def _call_openai(self, request: LLMRequest) -> LLMResponse:
        """Call OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.config['openai_api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop": request.stop_sequences,
            "stream": request.stream
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/completions",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                
                return LLMResponse(
                    text=data["choices"][0]["text"],
                    usage=data["usage"],
                    model=data["model"],
                    finish_reason=data["choices"][0]["finish_reason"]
                )
    
    async def _call_anthropic(self, request: LLMRequest) -> LLMResponse:
        """Call Anthropic Claude API"""
        headers = {
            "x-api-key": self.config['anthropic_api_key'],
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": request.model,
            "prompt": f"\n\nHuman: {request.prompt}\n\nAssistant:",
            "max_tokens_to_sample": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop_sequences": request.stop_sequences
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/complete",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                
                return LLMResponse(
                    text=data["completion"],
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    model=data["model"],
                    finish_reason=data["stop_reason"]
                )
```

### Prompt Engineering and Management
```python
# Advanced Prompt Management System
from typing import Dict, List, Any
import json
import hashlib
from datetime import datetime

class PromptTemplate:
    def __init__(self, name: str, template: str, variables: List[str], metadata: Dict[str, Any] = None):
        self.name = name
        self.template = template
        self.variables = variables
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.version = 1
    
    def render(self, **kwargs) -> str:
        """Render template with provided variables"""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")
    
    def validate_variables(self, **kwargs) -> bool:
        """Validate that all required variables are provided"""
        return all(var in kwargs for var in self.variables)

class PromptManager:
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.prompt_cache: Dict[str, str] = {}
        self.usage_stats: Dict[str, int] = {}
    
    def add_template(self, template: PromptTemplate):
        """Add a new prompt template"""
        self.templates[template.name] = template
    
    def get_template(self, name: str) -> PromptTemplate:
        """Get a prompt template by name"""
        if name not in self.templates:
            raise ValueError(f"Template {name} not found")
        return self.templates[name]
    
    def generate_prompt(self, template_name: str, **kwargs) -> str:
        """Generate a prompt using a template"""
        template = self.get_template(template_name)
        
        # Validate variables
        if not template.validate_variables(**kwargs):
            raise ValueError("Missing required variables")
        
        # Generate cache key
        cache_key = self._generate_cache_key(template_name, kwargs)
        
        # Check cache
        if cache_key in self.prompt_cache:
            self.usage_stats[template_name] = self.usage_stats.get(template_name, 0) + 1
            return self.prompt_cache[cache_key]
        
        # Generate prompt
        prompt = template.render(**kwargs)
        
        # Cache the result
        self.prompt_cache[cache_key] = prompt
        self.usage_stats[template_name] = self.usage_stats.get(template_name, 0) + 1
        
        return prompt
    
    def _generate_cache_key(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Generate a cache key for the prompt"""
        key_data = f"{template_name}:{json.dumps(variables, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for all templates"""
        return self.usage_stats.copy()
    
    def clear_cache(self):
        """Clear the prompt cache"""
        self.prompt_cache.clear()

# Example prompt templates
def create_example_templates():
    manager = PromptManager()
    
    # Code generation template
    code_template = PromptTemplate(
        name="code_generation",
        template="""Generate {language} code for the following task:

Task: {task_description}
Requirements: {requirements}
Code style: {code_style}

Please provide:
1. Complete, working code
2. Comments explaining the logic
3. Error handling
4. Test cases if applicable

Code:""",
        variables=["language", "task_description", "requirements", "code_style"],
        metadata={"category": "code", "difficulty": "intermediate"}
    )
    
    # Text summarization template
    summary_template = PromptTemplate(
        name="text_summarization",
        template="""Summarize the following text in {max_length} words or less:

Text: {text}

Summary:""",
        variables=["max_length", "text"],
        metadata={"category": "summarization", "difficulty": "easy"}
    )
    
    # Question answering template
    qa_template = PromptTemplate(
        name="question_answering",
        template="""Answer the following question based on the provided context:

Context: {context}
Question: {question}

Answer:""",
        variables=["context", "question"],
        metadata={"category": "qa", "difficulty": "intermediate"}
    )
    
    manager.add_template(code_template)
    manager.add_template(summary_template)
    manager.add_template(qa_template)
    
    return manager
```

## 🚀 Vector Databases and Embeddings

### Vector Database Integration
```python
# Vector Database Service
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import aiohttp
import json

class VectorDatabase:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = config.get('embedding_model', 'text-embedding-ada-002')
        self.dimension = config.get('dimension', 1536)
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
    
    async def create_collection(self, name: str, dimension: int = None) -> bool:
        """Create a new vector collection"""
        dimension = dimension or self.dimension
        
        payload = {
            "name": name,
            "dimension": dimension,
            "metric": "cosine"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config['base_url']}/collections",
                headers=self._get_headers(),
                json=payload
            ) as response:
                return response.status == 201
    
    async def add_vectors(self, collection_name: str, vectors: List[Dict[str, Any]]) -> bool:
        """Add vectors to a collection"""
        payload = {
            "vectors": vectors
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config['base_url']}/collections/{collection_name}/vectors",
                headers=self._get_headers(),
                json=payload
            ) as response:
                return response.status == 201
    
    async def search(self, collection_name: str, query_vector: List[float], 
                    limit: int = 10, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        payload = {
            "vector": query_vector,
            "limit": limit,
            "filter": filter
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config['base_url']}/collections/{collection_name}/search",
                headers=self._get_headers(),
                json=payload
            ) as response:
                data = await response.json()
                return data.get('results', [])
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        return {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }

class EmbeddingService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def get_embedding(self, text: str, model: str = None) -> List[float]:
        """Get embedding for text"""
        model = model or self.config.get('default_model', 'text-embedding-ada-002')
        
        # Check cache
        cache_key = f"{model}:{hash(text)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate embedding
        embedding = await self._generate_embedding(text, model)
        
        # Cache the result
        self.cache[cache_key] = embedding
        
        return embedding
    
    async def get_embeddings_batch(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        model = model or self.config.get('default_model', 'text-embedding-ada-002')
        
        # Check cache for each text
        embeddings = []
        texts_to_process = []
        indices = []
        
        for i, text in enumerate(texts):
            cache_key = f"{model}:{hash(text)}"
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
            else:
                texts_to_process.append(text)
                indices.append(i)
                embeddings.append(None)
        
        # Process uncached texts
        if texts_to_process:
            batch_embeddings = await self._generate_embeddings_batch(texts_to_process, model)
            
            for i, embedding in enumerate(batch_embeddings):
                cache_key = f"{model}:{hash(texts_to_process[i])}"
                self.cache[cache_key] = embedding
                embeddings[indices[i]] = embedding
        
        return embeddings
    
    async def _generate_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding for a single text"""
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": text,
            "model": model
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                return data["data"][0]["embedding"]
    
    async def _generate_embeddings_batch(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": texts,
            "model": model
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                return [item["embedding"] for item in data["data"]]
```

### RAG (Retrieval-Augmented Generation) System
```python
# RAG System Implementation
from typing import List, Dict, Any, Optional
import asyncio
import json
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float] = None

@dataclass
class RAGResponse:
    answer: str
    sources: List[Document]
    confidence: float
    metadata: Dict[str, Any] = None

class RAGSystem:
    def __init__(self, llm_service: LLMService, vector_db: VectorDatabase, 
                 embedding_service: EmbeddingService, prompt_manager: PromptManager):
        self.llm_service = llm_service
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.prompt_manager = prompt_manager
        self.collection_name = "documents"
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the RAG system"""
        # Generate embeddings for documents
        texts = [doc.content for doc in documents]
        embeddings = await self.embedding_service.get_embeddings_batch(texts)
        
        # Prepare vectors for database
        vectors = []
        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]
            vectors.append({
                "id": doc.id,
                "vector": doc.embedding,
                "metadata": {
                    "content": doc.content,
                    **doc.metadata
                }
            })
        
        # Add to vector database
        return await self.vector_db.add_vectors(self.collection_name, vectors)
    
    async def query(self, question: str, max_sources: int = 5, 
                   model_provider: ModelProvider = ModelProvider.OPENAI) -> RAGResponse:
        """Query the RAG system"""
        # Generate query embedding
        query_embedding = await self.embedding_service.get_embedding(question)
        
        # Search for relevant documents
        search_results = await self.vector_db.search(
            self.collection_name,
            query_embedding,
            limit=max_sources
        )
        
        # Extract documents
        sources = []
        for result in search_results:
            doc = Document(
                id=result["id"],
                content=result["metadata"]["content"],
                metadata=result["metadata"],
                embedding=result["vector"]
            )
            sources.append(doc)
        
        # Generate context
        context = "\n\n".join([doc.content for doc in sources])
        
        # Generate answer using LLM
        prompt = self.prompt_manager.generate_prompt(
            "question_answering",
            context=context,
            question=question
        )
        
        llm_request = LLMRequest(
            prompt=prompt,
            model="gpt-3.5-turbo",
            max_tokens=500,
            temperature=0.7
        )
        
        llm_response = await self.llm_service.generate(llm_request, model_provider)
        
        # Calculate confidence based on source similarity
        confidence = self._calculate_confidence(search_results)
        
        return RAGResponse(
            answer=llm_response.text,
            sources=sources,
            confidence=confidence,
            metadata={
                "model": llm_response.model,
                "usage": llm_response.usage
            }
        )
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence based on search results"""
        if not search_results:
            return 0.0
        
        # Use the highest similarity score as confidence
        similarities = [result.get("similarity", 0.0) for result in search_results]
        return max(similarities) if similarities else 0.0
```

## 🔧 Advanced AI Backend Patterns

### Model Fine-tuning and Management
```python
# Model Fine-tuning Service
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
import json
from datetime import datetime
from enum import Enum

class FineTuningStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class FineTuningJob:
    def __init__(self, job_id: str, model_name: str, training_data: List[Dict[str, str]], 
                 hyperparameters: Dict[str, Any] = None):
        self.job_id = job_id
        self.model_name = model_name
        self.training_data = training_data
        self.hyperparameters = hyperparameters or {}
        self.status = FineTuningStatus.PENDING
        self.created_at = datetime.utcnow()
        self.completed_at = None
        self.fine_tuned_model = None
        self.metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_name": self.model_name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "fine_tuned_model": self.fine_tuned_model,
            "metrics": self.metrics
        }

class FineTuningService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.jobs: Dict[str, FineTuningJob] = {}
        self.monitoring_interval = 30  # seconds
    
    async def create_fine_tuning_job(self, model_name: str, training_data: List[Dict[str, str]], 
                                   hyperparameters: Dict[str, Any] = None) -> str:
        """Create a new fine-tuning job"""
        job_id = f"ft-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        job = FineTuningJob(
            job_id=job_id,
            model_name=model_name,
            training_data=training_data,
            hyperparameters=hyperparameters
        )
        
        self.jobs[job_id] = job
        
        # Start fine-tuning process
        asyncio.create_task(self._run_fine_tuning_job(job))
        
        return job_id
    
    async def _run_fine_tuning_job(self, job: FineTuningJob):
        """Run the fine-tuning job"""
        try:
            job.status = FineTuningStatus.RUNNING
            
            # Prepare training data
            training_file = await self._prepare_training_data(job.training_data)
            
            # Start fine-tuning
            fine_tuning_id = await self._start_fine_tuning(job.model_name, training_file, job.hyperparameters)
            
            # Monitor progress
            await self._monitor_fine_tuning(job, fine_tuning_id)
            
        except Exception as e:
            job.status = FineTuningStatus.FAILED
            print(f"Fine-tuning job {job.job_id} failed: {e}")
    
    async def _prepare_training_data(self, training_data: List[Dict[str, str]]) -> str:
        """Prepare training data in the required format"""
        # Convert to JSONL format
        jsonl_data = "\n".join([json.dumps(item) for item in training_data])
        
        # Upload to file storage (implement based on your storage solution)
        file_id = await self._upload_training_file(jsonl_data)
        
        return file_id
    
    async def _start_fine_tuning(self, model_name: str, training_file: str, 
                               hyperparameters: Dict[str, Any]) -> str:
        """Start the fine-tuning process"""
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "training_file": training_file,
            "model": model_name,
            **hyperparameters
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/fine_tuning/jobs",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                return data["id"]
    
    async def _monitor_fine_tuning(self, job: FineTuningJob, fine_tuning_id: str):
        """Monitor the fine-tuning progress"""
        while job.status == FineTuningStatus.RUNNING:
            try:
                status = await self._get_fine_tuning_status(fine_tuning_id)
                
                if status["status"] == "succeeded":
                    job.status = FineTuningStatus.COMPLETED
                    job.fine_tuned_model = status["fine_tuned_model"]
                    job.completed_at = datetime.utcnow()
                    job.metrics = status.get("result_files", [])
                    break
                elif status["status"] == "failed":
                    job.status = FineTuningStatus.FAILED
                    break
                elif status["status"] == "cancelled":
                    job.status = FineTuningStatus.CANCELLED
                    break
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Error monitoring fine-tuning job {job.job_id}: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _get_fine_tuning_status(self, fine_tuning_id: str) -> Dict[str, Any]:
        """Get the status of a fine-tuning job"""
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_id}",
                headers=headers
            ) as response:
                return await response.json()
    
    async def _upload_training_file(self, data: str) -> str:
        """Upload training file to storage"""
        # Implement file upload logic
        # Return file ID
        pass
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a fine-tuning job"""
        if job_id not in self.jobs:
            return None
        
        return self.jobs[job_id].to_dict()
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all fine-tuning jobs"""
        return [job.to_dict() for job in self.jobs.values()]
```

## 🎯 Best Practices

### Performance Optimization
1. **Caching**: Implement multi-level caching for embeddings and responses
2. **Batching**: Process multiple requests in batches
3. **Streaming**: Use streaming for real-time responses
4. **Connection Pooling**: Reuse connections for API calls
5. **Rate Limiting**: Implement proper rate limiting

### Security Considerations
1. **API Key Management**: Secure storage and rotation of API keys
2. **Input Validation**: Validate all inputs to prevent injection attacks
3. **Output Filtering**: Filter outputs to prevent harmful content
4. **Access Control**: Implement proper authentication and authorization
5. **Audit Logging**: Log all AI-related activities

### Monitoring and Observability
1. **Metrics**: Track usage, performance, and costs
2. **Logging**: Implement structured logging
3. **Alerting**: Set up alerts for failures and anomalies
4. **Tracing**: Implement distributed tracing
5. **Dashboards**: Create monitoring dashboards

---

**Last Updated**: December 2024  
**Category**: Generative AI Backend Systems  
**Complexity**: Expert Level
