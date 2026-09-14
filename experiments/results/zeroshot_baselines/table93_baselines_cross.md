| Method | Family | Training data | R@3 | R@5 | nDCG@5 |
|---|---|---|--:|--:|--:|
| **Computer Science** | | | | | |
| BM25 | Sparse lexical | None | 0.1869 | 0.2319 | 0.2966 |
| BGE-large | Dense embedding | External English retrieval pairs | 0.2037 | 0.2644 | 0.3285 |
| Qwen3-Embedding | Dense embedding | External multilingual relevance pairs | 0.2200 | 0.2849 | 0.3511 |
| SPECTER2-base | Dense embedding | Citation-graph contrastive (scientific) | 0.1772 | 0.2353 | 0.2933 |
| SciNCL | Dense embedding | Citation-neighbourhood contrastive (scientific) | 0.1766 | 0.2300 | 0.2891 |
| ReasonIR-8B | Reasoning-trained dense | Synthetic reasoning-intensive retrieval pairs | 0.2137 | 0.2643 | 0.3322 |
| | | *n = 247 cross-field queries* | | | |
| **Materials Science** | | | | | |
| BM25 | Sparse lexical | None | 0.2708 | 0.3293 | 0.3871 |
| BGE-large | Dense embedding | External English retrieval pairs | 0.3083 | 0.3792 | 0.4299 |
| Qwen3-Embedding | Dense embedding | External multilingual relevance pairs | 0.3075 | 0.4012 | 0.4406 |
| SPECTER2-base | Dense embedding | Citation-graph contrastive (scientific) | 0.2607 | 0.3110 | 0.3618 |
| SciNCL | Dense embedding | Citation-neighbourhood contrastive (scientific) | 0.2443 | 0.2902 | 0.3396 |
| ReasonIR-8B | Reasoning-trained dense | Synthetic reasoning-intensive retrieval pairs | 0.3001 | 0.3847 | 0.4232 |
| | | *n = 121 cross-field queries* | | | |
