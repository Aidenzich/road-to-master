# What is memory?
| Type | Human| LLM |
| ------ | ------- | --- |
| Sensory Memory | This is the earliest stage of memory, providing the ability to retain impressions of sensory information (visual, auditory, etc) after the original stimuli have ended. Sensory memory typically only lasts for up to a few seconds. Subcategories include iconic memory (visual), echoic memory (auditory), and haptic memory (touch). | Sensory memory as learning embedding representations for raw inputs, including text, image or other modalities; |
| Short-Term Memory (STM)| It stores information that we are currently aware of and needed to carry out complex cognitive tasks such as learning and reasoning. Short-term memory is believed to have the capacity of about 7 items (Miller 1956) and lasts for 20-30 seconds. | Short-term memory as in-context learning. It is short and finite, as it is restricted by the finite context window length of Transformer.|
| Long-Term Memory (LTM) | Long-term memory can store information for a remarkably long time, ranging from a few days to decades, with an essentially unlimited storage capacity. There are two subtypes of LTM: | Long-term memory as the external vector store that the agent can attend to at query time, accessible via fast retrieval. |
|| - Explicit / declarative memory: This is memory of facts and events, and refers to those memories that can be consciously recalled, including episodic memory (events and experiences) and semantic memory (facts and concepts). ||
|| - Implicit / procedural memory: This type of memory is unconscious and involves skills and routines that are performed automatically, like riding a bike or typing on a keyboard. ||


## Vector Database 
| Database | Developed By | Description | Standout Features |
|------|----|-----------------|-------------------|
| **Faiss**| Facebook AI Research | Efficient similarity search and clustering of dense vectors.| GPU/CPU support, various indexing methods (LSH, PQ, HNSW), clustering algorithms (k-means, PCA), Python/C++ interfaces. |
| **Annoy**| Spotify| Approximate nearest neighbor search. Memory-efficient and optimized for quick queries. | Static/dynamic datasets, memory-mapped files, custom distance metrics, Python/C++ bindings.|
| **Milvus** | -| Designed for AI and analytics. Emphasis on massive high-dimensional data. | GPU/CPU support, various indexing methods, hybrid search, extensive SDK support (Python, Java, Node.js, C++), scalability.|
| **HNSWLIB**| -| Implements Hierarchical Navigable Small World (HNSW) algorithm. Emphasizing high-performance similarity search. | Fast query performance, multi-core parallelism, custom distance metrics, Python/C++ interfaces. |
| **NMSLIB** | -| Similarity and nearest neighbor search in both metric and non-metric spaces.| Range of indexing methods, custom distance metrics, efficient processing, bindings in Python, C++, and Java. |
| **Cottontail DB**| -| Multimedia retrieval column store. Both boolean and vector-space retrieval. | Various indexing methods, wide range of distance functions, arithmetic vector operations, free-text search, gRPC interface.|
