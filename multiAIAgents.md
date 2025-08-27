## 🏗️ **System Architecture Deep Dive**

### **1. Core Design Patterns Used**

#### **A. Singleton Pattern**
```python
# Global instances - single point of access
multi_api_key_manager = MultiAPIKeyManager()
async_azure_openai_service = AsyncAzureOpenAIService()
```

**Why?** Ensures only one instance manages API keys and service state across the entire application.

#### **B. Factory Pattern**
```python
async def create_llm_with_key(self, api_key_config: APIKeyConfig) -> AzureChatOpenAI:
    """Create a LangChain Azure OpenAI LLM instance with specific API key"""
    return AzureChatOpenAI(
        azure_deployment=api_key_config.deployment_name,
        openai_api_version=api_key_config.api_version,
        azure_endpoint=api_key_config.endpoint,
        api_key=api_key_config.api_key,
        # ... other config
    )
```

**Why?** Creates different LLM instances based on different API key configurations.

#### **C. Strategy Pattern**
```python
# Different processing strategies
async def _execute_crewai_with_retry(self, config, context, max_retries=2):
    # Strategy 1: Initial requests (2 retries, 2-second delays)
    # Strategy 2: Retry requests (3 retries, 1-second delays)
    retry_delay = 1 if is_retry_request else 2
```

**Why?** Different retry strategies based on request type.

#### **D. Observer Pattern**
```python
# Callbacks for monitoring
self.on_request_completed: Optional[Callable] = None
self.on_request_failed: Optional[Callable] = None
```

**Why?** Notify other parts of the system when requests complete or fail.

#### **E. Semaphore Pattern (Concurrency Control)**
```python
self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)

async with self.request_semaphore:
    # Only max_concurrent_requests can execute simultaneously
    self.active_requests += 1
    # ... process request
    self.active_requests -= 1
```

**Why?** Limits concurrent requests to prevent overwhelming the system.

## 🔍 **Debugging Techniques & Tools**

### **1. Comprehensive Logging Strategy**

Let me show you how to add better debugging:

Ran tool
Let me create the debug logger using a different approach:

Search files...
Perfect! Now let me teach you the complete system architecture and debugging techniques:

## 🎓 **Complete System Architecture & Debugging Guide**

### **1. System Architecture Overview**

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Frontend  │  │   Mobile    │  │   API       │            │
│  │   (React)   │  │   (Flutter) │  │   Clients   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API GATEWAY LAYER                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              FastAPI Server (Uvicorn)                      │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │   Routes    │  │ Middleware  │  │   Auth      │        │ │
│  │  │ (Endpoints) │  │ (CORS, etc) │  │ (JWT)       │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SERVICE LAYER                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Microtopic  │  │   Multi     │  │   Async     │            │
│  │  Service    │  │   API Key   │  │   Azure     │            │
│  │             │  │  Manager    │  │   Service   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│         │                │                │                   │
│         └────────────────┼────────────────┘                   │
│                          │                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   CrewAI    │  │   Cache     │  │   Database  │            │
│  │ Orchestrator│  │   Service   │  │   Service   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ PostgreSQL  │  │   Redis     │  │   Azure     │            │
│  │  Database   │  │   Cache     │  │   OpenAI    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### **2. Key Design Patterns Explained**

#### **A. Singleton Pattern (Global Instances)**
```python
# Why Singleton?
# 1. Single source of truth for API keys
# 2. Shared state across all requests
# 3. Memory efficiency

multi_api_key_manager = MultiAPIKeyManager()
async_azure_openai_service = AsyncAzureOpenAIService()
```

**Debugging Singleton Issues:**
```python
# Check if singleton is properly initialized
print(f"API Keys loaded: {len(multi_api_key_manager.api_keys)}")
print(f"Service max concurrent: {async_azure_openai_service.max_concurrent_requests}")

# Check singleton state
stats = await multi_api_key_manager.get_key_stats()
print(f"Active keys: {stats['active_keys']}")
```

#### **B. Semaphore Pattern (Concurrency Control)**
```python
# Why Semaphore?
# 1. Limits concurrent requests to prevent overload
# 2. Prevents resource exhaustion
# 3. Provides backpressure

self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)

async with self.request_semaphore:
    # Only max_concurrent_requests can execute here simultaneously
    self.active_requests += 1
    # ... process request
    self.active_requests -= 1
```

**Debugging Semaphore Issues:**
```python
# Check semaphore state
print(f"Semaphore available: {self.request_semaphore._value}")
print(f"Active requests: {self.active_requests}")

# Monitor semaphore usage
async def debug_semaphore_usage():
    while True:
        print(f"Semaphore: {self.request_semaphore._value}/{self.max_concurrent_requests}")
        await asyncio.sleep(5)
```

#### **C. Round-Robin Load Balancing**
```python
# Why Round-Robin?
# 1. Distributes load evenly across API keys
# 2. Simple and predictable
# 3. Easy to debug

selected_key = active_keys[self.current_key_index % len(active_keys)]
self.current_key_index += 1
```

**Debugging Load Balancing:**
```python
# Track key usage
for key_id, key in self.api_keys.items():
    print(f"Key {key_id}: {key.usage_count} requests")

# Check distribution
total_requests = sum(key.usage_count for key in self.api_keys.values())
for key_id, key in self.api_keys.items():
    percentage = (key.usage_count / total_requests) * 100 if total_requests > 0 else 0
    print(f"Key {key_id}: {percentage:.1f}%")
```

### **3. Debugging Techniques & Tools**

#### **A. Request Lifecycle Tracking**
```python
# Add this to your service methods
import uuid
import time

class RequestTracker:
    def __init__(self):
        self.requests = {}
    
    def start_request(self, request_type: str):
        request_id = str(uuid.uuid4())
        self.requests[request_id] = {
            "type": request_type,
            "start_time": time.time(),
            "steps": [],
            "status": "started"
        }
        return request_id
    
    def add_step(self, request_id: str, step: str, details: dict = None):
        if request_id in self.requests:
            self.requests[request_id]["steps"].append({
                "step": step,
                "timestamp": time.time(),
                "details": details
            })
    
    def complete_request(self, request_id: str, success: bool, result: dict = None):
        if request_id in self.requests:
            self.requests[request_id].update({
                "status": "completed" if success else "failed",
                "end_time": time.time(),
                "result": result
            })
    
    def get_request_summary(self, request_id: str):
        return self.requests.get(request_id)

# Usage in your service
tracker = RequestTracker()

async def generate_content_async(self, request):
    request_id = tracker.start_request("content_generation")
    
    try:
        tracker.add_step(request_id, "api_key_selection")
        api_key = await self.get_available_api_key()
        
        tracker.add_step(request_id, "crewai_execution")
        result = await self.execute_crewai(config, context)
        
        tracker.complete_request(request_id, True, result)
        return result
        
    except Exception as e:
        tracker.complete_request(request_id, False, {"error": str(e)})
        raise
```

#### **B. Performance Monitoring**
```python
import time
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
    
    def start_operation(self, operation_name: str):
        self.start_times[operation_name] = time.time()
    
    def end_operation(self, operation_name: str):
        if operation_name in self.start_times:
            duration = time.time() - self.start_times[operation_name]
            self.metrics[operation_name].append(duration)
            del self.start_times[operation_name]
    
    def get_stats(self, operation_name: str):
        if operation_name not in self.metrics:
            return None
        
        durations = self.metrics[operation_name]
        return {
            "count": len(durations),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "recent_durations": durations[-10:]  # Last 10 operations
        }
    
    def print_summary(self):
        print("📊 Performance Summary:")
        for operation, durations in self.metrics.items():
            stats = self.get_stats(operation)
            print(f"  {operation}: {stats['count']} ops, avg: {stats['avg_duration']:.2f}s")

# Usage
monitor = PerformanceMonitor()

async def execute_with_monitoring(self, config, context):
    monitor.start_operation("crewai_execution")
    try:
        result = await self.orchestrator._execute_crewai_with_config(config, context)
        monitor.end_operation("crewai_execution")
        return result
    except Exception as e:
        monitor.end_operation("crewai_execution")
        raise
```

#### **C. API Key Health Monitoring**
```python
class APIKeyHealthMonitor:
    def __init__(self):
        self.health_checks = {}
    
    async def check_key_health(self, key_id: str, api_key_config: APIKeyConfig):
        """Test if an API key is working"""
        try:
            # Create a simple test request
            client = AzureOpenAI(
                api_version=api_key_config.api_version,
                azure_endpoint=api_key_config.endpoint,
                api_key=api_key_config.api_key,
                timeout=10.0,
            )
            
            response = await asyncio.to_thread(
                client.chat.completions.create,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                model=api_key_config.deployment_name,
            )
            
            self.health_checks[key_id] = {
                "status": "healthy",
                "last_check": time.time(),
                "response_time": time.time() - self.health_checks.get(key_id, {}).get("last_check", time.time())
            }
            
        except Exception as e:
            self.health_checks[key_id] = {
                "status": "unhealthy",
                "last_check": time.time(),
                "error": str(e)
            }
    
    async def check_all_keys(self, api_keys: dict):
        """Check health of all API keys"""
        tasks = []
        for key_id, config in api_keys.items():
            task = asyncio.create_task(self.check_key_health(key_id, config))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_unhealthy_keys(self):
        """Get list of unhealthy keys"""
        return [
            key_id for key_id, health in self.health_checks.items()
            if health.get("status") == "unhealthy"
        ]
    
    def print_health_summary(self):
        """Print health summary of all keys"""
        print("🔍 API Key Health Summary:")
        for key_id, health in self.health_checks.items():
            status_emoji = "✅" if health["status"] == "healthy" else "❌"
            print(f"  {status_emoji} {key_id}: {health['status']}")
            if health.get("error"):
                print(f"    Error: {health['error']}")

# Usage
health_monitor = APIKeyHealthMonitor()
await health_monitor.check_all_keys(multi_api_key_manager.api_keys)
health_monitor.print_health_summary()
```

### **4. Common Debugging Scenarios**

#### **A. "No Available API Keys" Error**
```python
# Debug this issue
async def debug_api_key_availability():
    print("🔍 Debugging API Key Availability:")
    
    # Check all keys
    for key_id, key_config in multi_api_key_manager.api_keys.items():
        print(f"  Key {key_id}:")
        print(f"    Active: {key_config.is_active}")
        print(f"    Error count: {key_config.error_count}")
        print(f"    Usage count: {key_config.usage_count}")
        print(f"    Last used: {key_config.last_used}")
        
        # Check if key is available
        is_available = multi_api_key_manager._is_key_available(key_config)
        print(f"    Available: {is_available}")
        print()
    
    # Check active keys
    active_keys = [
        key for key in multi_api_key_manager.api_keys.values()
        if key.is_active and multi_api_key_manager._is_key_available(key)
    ]
    print(f"Total available keys: {len(active_keys)}")
```

#### **B. "Service at Maximum Capacity" Error**
```python
# Debug capacity issues
async def debug_capacity_issues():
    print("�� Debugging Capacity Issues:")
    
    stats = await async_azure_openai_service.get_service_stats()
    print(f"Active requests: {stats['active_requests']}")
    print(f"Max concurrent: {stats['max_concurrent_requests']}")
    print(f"Available slots: {stats['available_slots']}")
    
    # Check if requests are stuck
    if stats['active_requests'] > 0:
        print("⚠️ There are active requests. Checking if they're stuck...")
        # Add logic to check request timeouts
```

#### **C. "All API Keys Failed" Error**
```python
# Debug API key failures
async def debug_api_key_failures():
    print("🔍 Debugging API Key Failures:")
    
    # Check error counts
    for key_id, key_config in multi_api_key_manager.api_keys.items():
        if key_config.error_count > 0:
            print(f"❌ Key {key_id} has {key_config.error_count} errors")
    
    # Test each key
    for key_id, key_config in multi_api_key_manager.api_keys.items():
        print(f"🧪 Testing key {key_id}...")
        try:
            result = await async_azure_openai_service.execute_with_api_key(
                key_config, 
                [{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            print(f"  ✅ Key {key_id} works")
        except Exception as e:
            print(f"  ❌ Key {key_id} failed: {e}")
```

### **5. Production Debugging Tools**

#### **A. Real-time Monitoring Dashboard**
```python
# Add this endpoint to your routes
@microtopic_content_router.get("/debug/dashboard")
async def get_debug_dashboard():
    """Get comprehensive debug information"""
    
    # Get all stats
    service_stats = await async_azure_openai_service.get_service_stats()
    api_key_stats = await multi_api_key_manager.get_key_stats()
    
    # Get performance metrics
    performance_stats = monitor.print_summary() if 'monitor' in globals() else {}
    
    # Get health status
    health_status = health_monitor.health_checks if 'health_monitor' in globals() else {}
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "service_stats": service_stats,
        "api_key_stats": api_key_stats,
        "performance_stats": performance_stats,
        "health_status": health_status,
        "system_info": {
            "python_version": sys.version,
            "asyncio_loop": asyncio.get_event_loop().is_running(),
            "memory_usage": psutil.virtual_memory().percent if 'psutil' in globals() else None
        }
    }
```

#### **B. Request Tracing**
```python
# Add request tracing to your endpoints
import traceback

@microtopic_content_router.post("/generate-async")
async def generate_content_async(request: MicrotopicRequest, ...):
    request_id = str(uuid.uuid4())
    
    # Start tracing
    trace = {
        "request_id": request_id,
        "start_time": time.time(),
        "steps": [],
        "errors": []
    }
    
    try:
        # Step 1: Validate request
        trace["steps"].append({
            "step": "request_validation",
            "timestamp": time.time(),
            "details": {"subject": request.subject_name, "topic": request.topic_name}
        })
        
        # Step 2: Check capacity
        stats = await async_azure_openai_service.get_service_stats()
        trace["steps"].append({
            "step": "capacity_check",
            "timestamp": time.time(),
            "details": stats
        })
        
        # Step 3: Generate content
        result = await microtopic_content_service.generate_lite_microtopic_content(request)
        trace["steps"].append({
            "step": "content_generation",
            "timestamp": time.time(),
            "details": {"success": result.success}
        })
        
        # Add trace to response
        if not result.metadata:
            result.metadata = {}
        result.metadata["trace"] = trace
        
        return result
        
    except Exception as e:
        trace["errors"].append({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": time.time()
        })
        
        # Log trace for debugging
        logger.error(f"Request {request_id} failed: {trace}")
        raise
```

### **6. Debugging Checklist**

When debugging parallel processing issues, follow this checklist:

#### **✅ Pre-debugging Checks**
1. **Check API Keys**: Are all keys valid and active?
2. **Check Capacity**: Is the service at maximum capacity?
3. **Check Network**: Are Azure endpoints reachable?
4. **Check Database**: Are database connections working?

#### **✅ Runtime Debugging**
1. **Monitor Logs**: Check for error patterns
2. **Track Requests**: Use request tracing
3. **Monitor Performance**: Check response times
4. **Check Health**: Monitor API key health

#### **✅ Post-debugging Analysis**
1. **Analyze Patterns**: Look for common failure points
2. **Optimize Configuration**: Adjust timeouts, retries, etc.
3. **Update Monitoring**: Add alerts for common issues
4. **Document Solutions**: Keep track of fixes

### **7. Quick Debug Commands**

```bash
# Check service status
curl -X GET "http://localhost:8001/api/v1/microtopic-content/service-stats"

# Test single request
curl -X POST "http://localhost:8001/api/v1/microtopic-content/generate-async" \
  -H "Content-Type: application/json" \
  -d '{"course_id":"test","subject_name":"Math","topic_name":"Algebra"}'

# Check debug dashboard
curl -X GET "http://localhost:8001/api/v1/microtopic-content/debug/dashboard"

# Monitor logs in real-time
tail -f debug_parallel_processing.log
```
