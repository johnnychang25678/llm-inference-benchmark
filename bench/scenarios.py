"""Benchmark scenario definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    """A benchmark prompt."""

    name: str
    system: str
    user: str

    def to_completion_prompt(self) -> str:
        """Convert to completion-style prompt.

        Returns:
            Formatted prompt string.
        """
        return f"System: {self.system}\n\nUser: {self.user}\n\nAssistant:"


# Built-in prompts for different sizes
PROMPTS = {
    "small": Prompt(
        name="small",
        system="You are a helpful assistant.",
        user="Say hello in exactly 10 words.",
    ),
    "medium": Prompt(
        name="medium",
        system="You are a helpful assistant who explains concepts clearly and concisely.",
        user=(
            "Explain the concept of machine learning in 3 short paragraphs. "
            "Cover what it is, how it works, and one real-world application."
        ),
    ),
    "large": Prompt(
        name="large",
        system=(
            "You are a senior software engineer with expertise in distributed systems, "
            "cloud architecture, and performance optimization. You provide detailed, "
            "technical explanations with code examples when appropriate."
        ),
        user=(
            "Design a high-availability caching system for a web application that "
            "serves 100,000 requests per second. The system should handle:\n\n"
            "1. Cache invalidation strategies\n"
            "2. Data consistency across multiple cache nodes\n"
            "3. Failover mechanisms\n"
            "4. Memory management and eviction policies\n\n"
            "Provide a detailed architecture with specific technology choices and "
            "explain the trade-offs of your design decisions. Include pseudocode "
            "for the critical cache operations."
        ),
    ),
    "xlarge": Prompt(
        name="xlarge",
        system=(
            "You are a world-class software architect with 20+ years of experience "
            "designing systems at Google, Amazon, and Netflix scale. You have deep "
            "expertise in distributed systems, microservices, event-driven architecture, "
            "database design, caching strategies, and observability. Your explanations "
            "are thorough, technically precise, and include real-world trade-offs. "
            "You always provide concrete code examples in Python or Go when relevant."
        ),
        user=(
            "Design a complete real-time analytics platform that can process 1 million "
            "events per second with sub-second query latency. The platform should:\n\n"
            "1. **Data Ingestion Layer**:\n"
            "   - Handle multiple data sources (Kafka, REST APIs, WebSockets)\n"
            "   - Implement backpressure and rate limiting\n"
            "   - Support schema evolution and data validation\n\n"
            "2. **Stream Processing**:\n"
            "   - Real-time aggregations (counts, sums, averages, percentiles)\n"
            "   - Windowed computations (tumbling, sliding, session windows)\n"
            "   - Exactly-once processing guarantees\n"
            "   - Handle late-arriving data\n\n"
            "3. **Storage Layer**:\n"
            "   - Hot/warm/cold data tiering strategy\n"
            "   - Time-series optimized storage\n"
            "   - Efficient compression and encoding\n"
            "   - Data retention and archival policies\n\n"
            "4. **Query Engine**:\n"
            "   - Support for SQL-like queries\n"
            "   - Real-time and historical data joins\n"
            "   - Query optimization and caching\n"
            "   - Concurrent query handling\n\n"
            "5. **Observability & Operations**:\n"
            "   - Monitoring and alerting strategy\n"
            "   - Disaster recovery and failover\n"
            "   - Capacity planning and auto-scaling\n\n"
            "Provide detailed architecture diagrams (in ASCII), technology choices with "
            "justification, code examples for critical components, and discuss the "
            "trade-offs of your design decisions. Include estimated resource requirements "
            "and cost considerations."
        ),
    ),
}

# Varying prompts to test without cache benefits (different user prompts)
VARYING_PROMPTS = {
    "small": [
        Prompt(
            name="small_v1",
            system="You are a helpful assistant.",
            user="Say hello in exactly 10 words.",
        ),
        Prompt(
            name="small_v2",
            system="You are a helpful assistant.",
            user="Count from 1 to 10 in words.",
        ),
        Prompt(
            name="small_v3",
            system="You are a helpful assistant.",
            user="Name 5 colors of the rainbow.",
        ),
        Prompt(
            name="small_v4",
            system="You are a helpful assistant.",
            user="List the days of the week.",
        ),
        Prompt(
            name="small_v5",
            system="You are a helpful assistant.",
            user="Name the four seasons.",
        ),
        Prompt(
            name="small_v6",
            system="You are a helpful assistant.",
            user="List five common fruits.",
        ),
        Prompt(
            name="small_v7",
            system="You are a helpful assistant.",
            user="Name the planets in order.",
        ),
        Prompt(
            name="small_v8",
            system="You are a helpful assistant.",
            user="What are the primary colors?",
        ),
        Prompt(
            name="small_v9",
            system="You are a helpful assistant.",
            user="List the months of the year.",
        ),
        Prompt(
            name="small_v10",
            system="You are a helpful assistant.",
            user="Name five ocean animals.",
        ),
    ],
    "medium": [
        Prompt(
            name="medium_v1",
            system="You are a helpful assistant who explains concepts clearly.",
            user="Explain machine learning in 3 short paragraphs.",
        ),
        Prompt(
            name="medium_v2",
            system="You are a helpful assistant who explains concepts clearly.",
            user="Explain how the internet works in 3 short paragraphs.",
        ),
        Prompt(
            name="medium_v3",
            system="You are a helpful assistant who explains concepts clearly.",
            user="Explain how airplanes fly in 3 short paragraphs.",
        ),
        Prompt(
            name="medium_v4",
            system="You are a helpful assistant who explains concepts clearly.",
            user="Explain photosynthesis in 3 short paragraphs.",
        ),
        Prompt(
            name="medium_v5",
            system="You are a helpful assistant who explains concepts clearly.",
            user="Explain how vaccines work in 3 short paragraphs.",
        ),
        Prompt(
            name="medium_v6",
            system="You are a helpful assistant who explains concepts clearly.",
            user="Explain the water cycle in 3 short paragraphs.",
        ),
        Prompt(
            name="medium_v7",
            system="You are a helpful assistant who explains concepts clearly.",
            user="Explain how electricity works in 3 short paragraphs.",
        ),
        Prompt(
            name="medium_v8",
            system="You are a helpful assistant who explains concepts clearly.",
            user="Explain the theory of evolution in 3 short paragraphs.",
        ),
        Prompt(
            name="medium_v9",
            system="You are a helpful assistant who explains concepts clearly.",
            user="Explain how GPS navigation works in 3 short paragraphs.",
        ),
        Prompt(
            name="medium_v10",
            system="You are a helpful assistant who explains concepts clearly.",
            user="Explain how the human immune system works in 3 short paragraphs.",
        ),
    ],
    "large": [
        Prompt(
            name="large_v1",
            system=(
                "You are a senior software engineer with expertise in distributed systems, "
                "cloud architecture, and performance optimization. You provide detailed, "
                "technical explanations with code examples when appropriate."
            ),
            user=(
                "Design a high-availability caching system for a web application that "
                "serves 100,000 requests per second. The system should handle:\n\n"
                "1. Cache invalidation strategies (TTL, event-based, write-through)\n"
                "2. Data consistency across multiple cache nodes\n"
                "3. Failover mechanisms and health checking\n"
                "4. Memory management and eviction policies (LRU, LFU, ARC)\n\n"
                "Provide a detailed architecture with specific technology choices "
                "(Redis, Memcached, etc.) and explain the trade-offs. Include pseudocode "
                "for cache read/write operations with proper error handling."
            ),
        ),
        Prompt(
            name="large_v2",
            system=(
                "You are a senior software engineer with expertise in distributed systems, "
                "cloud architecture, and performance optimization. You provide detailed, "
                "technical explanations with code examples when appropriate."
            ),
            user=(
                "Design a distributed rate limiting system for an API gateway handling "
                "50,000 requests per second. The system should support:\n\n"
                "1. Multiple rate limiting algorithms (token bucket, sliding window, leaky bucket)\n"
                "2. Per-user, per-API-key, and global rate limits\n"
                "3. Distributed state synchronization across multiple gateway instances\n"
                "4. Graceful degradation and circuit breaker patterns\n\n"
                "Provide architecture details with Redis or similar backing store, "
                "explain the trade-offs between consistency and performance, and include "
                "pseudocode for the core rate checking logic."
            ),
        ),
        Prompt(
            name="large_v3",
            system=(
                "You are a senior software engineer with expertise in distributed systems, "
                "cloud architecture, and performance optimization. You provide detailed, "
                "technical explanations with code examples when appropriate."
            ),
            user=(
                "Design an event sourcing system for a financial trading application. "
                "The system should handle:\n\n"
                "1. Event storage with immutable append-only logs\n"
                "2. Snapshot strategies for fast aggregate reconstruction\n"
                "3. Multiple read-model projections for different query patterns\n"
                "4. Event replay and temporal queries\n\n"
                "Provide detailed architecture with technology choices (Kafka, EventStore, etc.), "
                "explain consistency guarantees and failure recovery, and include pseudocode "
                "for event application and projection updates."
            ),
        ),
        Prompt(
            name="large_v4",
            system=(
                "You are a senior software engineer with expertise in distributed systems, "
                "cloud architecture, and performance optimization. You provide detailed, "
                "technical explanations with code examples when appropriate."
            ),
            user=(
                "Design a distributed task queue system similar to Celery or Sidekiq. "
                "The system should support:\n\n"
                "1. Reliable task execution with at-least-once delivery guarantees\n"
                "2. Retry policies with exponential backoff and dead letter queues\n"
                "3. Task prioritization and fair scheduling across tenants\n"
                "4. Monitoring, alerting, and task introspection\n\n"
                "Provide detailed architecture with message broker choices (Redis, RabbitMQ, etc.), "
                "explain worker scaling strategies and failure handling, and include pseudocode "
                "for task submission and worker processing loops."
            ),
        ),
        Prompt(
            name="large_v5",
            system=(
                "You are a senior software engineer with expertise in distributed systems, "
                "cloud architecture, and performance optimization. You provide detailed, "
                "technical explanations with code examples when appropriate."
            ),
            user=(
                "Design a distributed search engine for an e-commerce platform with "
                "10 million products. The system should handle:\n\n"
                "1. Full-text search with relevance ranking and fuzzy matching\n"
                "2. Faceted search and filtering by multiple attributes\n"
                "3. Real-time index updates when products change\n"
                "4. Query auto-complete and search suggestions\n\n"
                "Provide detailed architecture with technology choices (Elasticsearch, Solr, etc.), "
                "explain indexing strategies and sharding approaches, and include pseudocode "
                "for the search query pipeline."
            ),
        ),
        Prompt(
            name="large_v6",
            system=(
                "You are a senior software engineer with expertise in distributed systems, "
                "cloud architecture, and performance optimization. You provide detailed, "
                "technical explanations with code examples when appropriate."
            ),
            user=(
                "Design a real-time notification system for a social media platform "
                "with 50 million active users. The system should support:\n\n"
                "1. Push notifications to mobile devices (iOS, Android)\n"
                "2. In-app notifications with read/unread status\n"
                "3. Email and SMS notification channels\n"
                "4. User preference management and rate limiting\n\n"
                "Provide detailed architecture with technology choices (Firebase, SNS, etc.), "
                "explain delivery guarantees and fan-out strategies, and include pseudocode "
                "for the notification dispatch pipeline."
            ),
        ),
        Prompt(
            name="large_v7",
            system=(
                "You are a senior software engineer with expertise in distributed systems, "
                "cloud architecture, and performance optimization. You provide detailed, "
                "technical explanations with code examples when appropriate."
            ),
            user=(
                "Design a distributed session management system for a multi-region "
                "web application. The system should handle:\n\n"
                "1. Session creation, validation, and expiration\n"
                "2. Cross-region session replication with low latency\n"
                "3. Session hijacking prevention and security measures\n"
                "4. Graceful handling of region failovers\n\n"
                "Provide detailed architecture with technology choices (Redis Cluster, DynamoDB, etc.), "
                "explain consistency vs availability trade-offs, and include pseudocode "
                "for session validation middleware."
            ),
        ),
        Prompt(
            name="large_v8",
            system=(
                "You are a senior software engineer with expertise in distributed systems, "
                "cloud architecture, and performance optimization. You provide detailed, "
                "technical explanations with code examples when appropriate."
            ),
            user=(
                "Design a distributed logging and monitoring system for a microservices "
                "architecture with 200 services. The system should support:\n\n"
                "1. Centralized log aggregation with structured logging\n"
                "2. Distributed tracing across service boundaries\n"
                "3. Real-time alerting based on log patterns and metrics\n"
                "4. Log retention policies and cost-effective storage\n\n"
                "Provide detailed architecture with technology choices (ELK, Datadog, etc.), "
                "explain sampling strategies and data pipeline design, and include pseudocode "
                "for the log ingestion pipeline."
            ),
        ),
        Prompt(
            name="large_v9",
            system=(
                "You are a senior software engineer with expertise in distributed systems, "
                "cloud architecture, and performance optimization. You provide detailed, "
                "technical explanations with code examples when appropriate."
            ),
            user=(
                "Design a distributed file storage system similar to S3 for storing "
                "user-uploaded media files. The system should handle:\n\n"
                "1. Multi-part uploads for large files\n"
                "2. Content deduplication and compression\n"
                "3. CDN integration for global content delivery\n"
                "4. Access control and pre-signed URLs\n\n"
                "Provide detailed architecture with technology choices (MinIO, Ceph, etc.), "
                "explain replication and durability guarantees, and include pseudocode "
                "for the upload and retrieval APIs."
            ),
        ),
        Prompt(
            name="large_v10",
            system=(
                "You are a senior software engineer with expertise in distributed systems, "
                "cloud architecture, and performance optimization. You provide detailed, "
                "technical explanations with code examples when appropriate."
            ),
            user=(
                "Design a distributed configuration management system for a large-scale "
                "microservices deployment. The system should support:\n\n"
                "1. Hierarchical configuration with environment overrides\n"
                "2. Real-time configuration updates without restarts\n"
                "3. Configuration versioning and rollback capabilities\n"
                "4. Secret management with encryption at rest\n\n"
                "Provide detailed architecture with technology choices (Consul, etcd, etc.), "
                "explain consistency guarantees and watch mechanisms, and include pseudocode "
                "for the configuration client SDK."
            ),
        ),
    ],
}


@dataclass
class Scenario:
    """A benchmark scenario configuration."""

    name: str
    prompts: list[Prompt]
    concurrency: int
    duration_seconds: float
    ollama_num_parallel: int = 1
    tenant_id: str = "benchmark"
    warmup_requests: int = 3


def get_matrix_scenarios(
    prompt_sizes: list[str],
    concurrency_levels: list[int],
    ollama_num_parallel_values: list[int] | None = None,
    duration_seconds: float = 30.0,
    vary_prompts: bool = False,
) -> list[Scenario]:
    """Generate benchmark matrix scenarios.

    Args:
        prompt_sizes: List of prompt size names (small, medium, large).
        concurrency_levels: List of concurrency levels.
        ollama_num_parallel_values: List of OLLAMA_NUM_PARALLEL values.
        duration_seconds: Duration for each scenario.
        vary_prompts: If True, use varying prompts to avoid cache benefits.

    Returns:
        List of scenarios for the full matrix.
    """
    if ollama_num_parallel_values is None:
        ollama_num_parallel_values = [1]

    scenarios = []

    for parallel in ollama_num_parallel_values:
        for size in prompt_sizes:
            if vary_prompts:
                prompts = VARYING_PROMPTS.get(size, [])
                if not prompts:
                    # Fall back to single prompt if no varying prompts defined
                    prompt = PROMPTS.get(size)
                    prompts = [prompt] if prompt else []
            else:
                prompt = PROMPTS.get(size)
                prompts = [prompt] if prompt else []

            if not prompts:
                continue

            for concurrency in concurrency_levels:
                name_suffix = "_vary" if vary_prompts else ""
                scenarios.append(
                    Scenario(
                        name=f"{size}_c{concurrency}_p{parallel}{name_suffix}",
                        prompts=prompts,
                        concurrency=concurrency,
                        duration_seconds=duration_seconds,
                        ollama_num_parallel=parallel,
                        tenant_id=f"bench_{size}",
                    )
                )

    return scenarios
