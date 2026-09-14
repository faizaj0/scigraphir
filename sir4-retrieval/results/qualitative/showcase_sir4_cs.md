# sir4_cs: showcase candidates (cross-field golds; cosine >= 25 and (graph <= 5 or fused <= 25); tier A = model top-10, B = graph top-5 only, C = rest)

730 golds with interpretations under 'SciGraphIR (frame graph + CCMP)'; 15 pass the filter; routes: Counter({'bridge': 8, 'none': 4, 'hub': 2, 'other': 1})

Read the top rows first. 'bridge' = the top route passes a function / limitation / method / finding frame; 'hub' = it only passes papers and a domain node (the failure signature); gates > 1 are hops CCMP amplified.

| # | score | tier | route | hops | cosine | scorer | graph | fused | frame graph, gate… | OpenIE graph | Qwen3-Emb. | gold | field | query |
|--:|--:|---|---|--:|--:|--:|--:|--:|---:|---:|---:|---|---|---|
| 1 | 8.5 | A | bridge | 2 | 31 | 32 | 5 | 9 | -- | -- | 31 | Otter: Generating Tests from Issues to Validate SWE Patches | Computer Science -> Engineering | Can programmers produce a runnable software system by only authoring a non-triv… |
| 2 | 8.4 | C | bridge | 4 | 71 | 19 | 147 | 18 | 21 | 29 | 73 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via… | artificial intelligence | How can vision-language models be trained to perform robust spatial reasoning a… |
| 3 | 8.3 | A | bridge | 4 | 66 | 7 | 184 | 6 | 9 | 16 | 63 | Recurrent World Models Facilitate Policy Evolution | Computer Science -> Social Sciences | What are the capabilities of current language-model-based agents in long-horizo… |
| 4 | 8.2 | C | bridge | 4 | 483 | 18 | 372 | 19 | 20 | 12 | >300 | Two-Person Cooperative Games | Computer Science -> Decision Sciences | How can a generative AI system and a Q&A forum achieve sustainable, mutually be… |
| 5 | 7.8 | C | bridge | 3 | 69 | 20 | 687 | 21 | 22 | 27 | 69 | On the dimension and entropy of probability distributions. | Computer Science -> Physics and Astrono… | How can joint-embedding predictive architectures be regularized to learn sparse… |
| 6 | 7.4 | C | bridge | 6 | 78 | 19 | 900 | 21 | 21 | 17 | 74 | Tail bounds for sums of geometric and exponential variables | Computer Science -> Mathematics | Is there any backoff protocol that is stable for a positive arrival rate? Prior… |
| 7 | 7.1 | C | bridge | 4 | 43 | 13 | 907 | 13 | 15 | 19 | 45 | Forecasting: principles and practice. | Computer Science -> Decision Sciences | How can intent drift, a gradual and subtle divergence of a network's operationa… |
| 8 | 7.0 | A | none | 0 | 111 | 2 | 5 | 2 | 2 | 1 | 123 | The Moral Machine experiment. | Computer Science -> Neuroscience | How do directed contextual influences in prompts shift moral triage decisions i… |
| 9 | 7.0 | A | none | 0 | 957 | 1 | 1965 | 1 | 1 | 1 | >300 | The effect of the introduction of reward upon the maze perf… | Computer Science -> Neuroscience | Can large language models exhibit latent learning, improving their performance… |
| 10 | 6.7 | C | bridge | 4 | 46 | 17 | 1015 | 19 | 19 | 17 | 43 | Wasserstein geometry of Gaussian measures | Computer Science -> Mathematics | How can we quantify the degree of fusion or separation between two latent repre… |
| 11 | 5.9 | B | hub | 3 | 81 | 80 | 1 | 23 | 20 | -- | 81 | Mathematical Analysis of Singularities in the Diffusion Mod… | Computer Science -> Engineering | Why does inverting a real image into the initial noise of a diffusion model pro… |
| 12 | 5.5 | B | hub | 3 | 29 | 11 | 4 | 11 | 11 | 13 | 30 | Policy Invariance Under Reward Transformations: Theory and… | Computer Science -> Economics, Economet… | How can we systematically design and verify reward functions for offline clinic… |
| 13 | 4.8 | A | none | 0 | 67 | 9 | 1144 | 9 | 9 | 15 | 65 | Statistical methodologies to pool across multiple intervent… | Computer Science -> Mathematics | How can the aggregation of locally trained models in federated learning be made… |
| 14 | 4.4 | B | other | 1 | 28 | 25 | 2 | 11 | 10 | -- | 28 | Gradient Descent Only Converges to Minimizers: Non-Isolated… | Computer Science -> Medicine | Can full-batch gradient descent, by reusing the training data across iterations… |
| 15 | 4.1 | C | none | 0 | 125 | 15 | 1196 | 17 | 16 | -- | 127 | Conditional logit analysis of qualitative choice behavior. | Computer Science -> Social Sciences | How can a conversational and longitudinal benchmark for stock recommendation be… |

## 1. 10.48550_arxiv.2602.13723  ->  Otter: Generating Tests from Issues to Validate SWE Patches

**Field:** Computer Science -> Engineering | **stratum:** cross | **golds for this query:** 4 | **route:** bridge, 2 hops (shortest 2)

**Query:** Can programmers produce a runnable software system by only authoring a non-trivial, multi-modal requirement document that describes hundreds of scenarios, despite the tendency of LLM-based code generation to degrade and hallucinate as requirements scale? Existing LLM-based code generation approaches parse natural language into code snippets, but their performance degrades significantly when requirements scale to multi-modal documents with hundreds of scenarios, often producing incorrect implementations or omitting crucial constraints. Multi-agent software development frameworks coordinate multiple LLM agents to simulate software teams, yet they do not anchor the generated code to a verifiab…

**Inspiration (gold):** Otter: Generating Tests from Issues to Validate SWE Patches. While there has been plenty of work on generating tests from existing code, there has been limited work on generating tests from issues. A correct test must validate the code patch that resolves the issue. This paper focuses on the scenario where that code patch does not yet exist. Doing so supports two major use-cases. First, it supports TDD (test-driven development), the discipline of "test first, write code later" that has well-doc…

**Ranks of this gold:** SciGraphIR 9, graph channel 5, multi-view scorer 32, Qwen3 cosine 31, Qwen3-Emb. 31

**Scorer views that matched best (hypothetical answers written for the query):**
- view 3 match 0.615: A model-based testing framework that generates exhaustive test cases from multi-modal requirements, employing combinatorial testing techniques to ensure all scenarios are covered before any code is generated.

**SciGraphIR (frame graph + CCMP)** (ranks: fused 9, graph 5, scorer 32, dense 31)
- w=17.38: [method] test-driven development (gate 1.00) --inv. builds on--> [method] otter (gate 1.00) --inv. contributes--> [paper] Otter: Generating Tests from Issues to Validate SWE Patches

**frame graph, gate off**: no interpretation

**OpenIE graph**: no interpretation

## 2. 10.48550_arxiv.2601.23251  ->  DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

**Field:** artificial intelligence | **stratum:** cross | **golds for this query:** 4 | **route:** bridge, 4 hops (shortest 3)

**Query:** How can vision-language models be trained to perform robust spatial reasoning and compositional understanding using limited data? Vision-language models are typically pre-trained on large-scale collections of web videos and image-text pairs, and fine-tuned on generic video question-answering datasets. Although they achieve high scores on standard benchmarks, they systematically fail on simple spatial reasoning, counting, and compositional understanding tasks, indicating that their performance stems from statistical pattern matching rather than grounded reasoning. Existing fine-tuning methods, such as supervised learning on instruction-following datasets, do not provide explicit supervision…

**Inspiration (gold):** DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. General reasoning represents a long-standing and formidable challenge in artificial intelligence. Recent breakthroughs, exemplified by large language models (LLMs) and chain-of-thought prompting, have achieved considerable success on foundational reasoning tasks. However, this success is heavily contingent upon extensive human-annotated demonstrations, and models' capabilities are still insufficient for more com…

**Ranks of this gold:** SciGraphIR 18, graph channel 147, multi-view scorer 19, Qwen3 cosine 71, frame graph, gate off|fused 21, frame graph, gate off|graph 1837, OpenIE graph|fused 29, OpenIE graph|graph 1832, Qwen3-Emb. 73

**Scorer views that matched best (hypothetical answers written for the query):**
- view 2 match 0.581: An interactive reinforcement learning algorithm that incorporates a structured feedback loop, allowing the model to learn spatial reasoning through simulated interactions with a visual environment that provides explicit reward signals for…

**SciGraphIR (frame graph + CCMP)** (ranks: fused 18, graph 147, scorer 19, dense 71)
- w=12.95: [method] visual question answering benchmarks (gate 1.00) --inv. builds on--> [method] clevr (gate 1.00) --inv. contributes--> [paper] CLEVR: A Diagnostic Dataset for Compositional Language and… (gate 1.00) --in field--> [domain] artificial intelligence (gate 1.18) --inv. in field--> [paper] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via…

**frame graph, gate off** (ranks: fused 21, graph 1837, scorer 19, dense 71)
- w=15.83: [method] visual question answering benchmarks (gate 1.00) --inv. builds on--> [method] clevr (gate 1.00) --inv. contributes--> [paper] CLEVR: A Diagnostic Dataset for Compositional Language and… (gate 1.00) --in field--> [domain] artificial intelligence (gate 1.18) --inv. in field--> [paper] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via…

**OpenIE graph** (ranks: fused 29, graph 1832, scorer 27, dense 71)
- w=0.04: reinforcement learning --inv. equivalent--> reinforcement learning rl --is mentioned in--> [paper] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via…

## 3. 10.48550_arxiv.2602.05843  ->  Recurrent World Models Facilitate Policy Evolution

**Field:** Computer Science -> Social Sciences | **stratum:** cross | **golds for this query:** 4 | **route:** bridge, 4 hops (shortest 3)

**Query:** What are the capabilities of current language-model-based agents in long-horizon active exploration and inductive inference of latent transition dynamics in interactive environments? Existing agent evaluation benchmarks primarily assess deductive reasoning by providing explicit rules and goals, often within short interaction horizons (fewer than 50 steps). These benchmarks typically supply full success criteria and detailed instructions, bypassing the need for active exploration and trial-and-error. Static reasoning benchmarks evaluate rule synthesis but are passive and do not capture the interactive discovery loop. As a result, no systematic method exists to measure an agent's capability t…

**Inspiration (gold):** Recurrent World Models Facilitate Policy Evolution. A generative recurrent neural network is quickly trained in an unsupervised manner to model popular reinforcement learning environments through compressed spatio-temporal representations. The world model's extracted features are fed into compact and simple policies trained by evolution, achieving state of the art results in various environments. We also train our agent entirely inside of an environment generated by its own internal world model…

**Ranks of this gold:** SciGraphIR 6, graph channel 184, multi-view scorer 7, Qwen3 cosine 66, frame graph, gate off|fused 9, frame graph, gate off|graph 2477, OpenIE graph|fused 16, OpenIE graph|graph 2566, Qwen3-Emb. 63

**Scorer views that matched best (hypothetical answers written for the query):**
- view 0 match 0.729: A long short-term memory (LSTM) network integrated with reinforcement learning techniques to model and predict latent state transitions in interactive environments based solely on agent experience and feedback.

**SciGraphIR (frame graph + CCMP)** (ranks: fused 6, graph 184, scorer 7, dense 66)
- w=13.69: [function] characterize learning dynamics at the parameter… (gate 1.00) --inv. achieves--> [method] rlvr (gate 1.00) --inv. contributes--> [paper] The Path Not Taken: RLVR Provably Learns Off the Principals (gate 1.00) --in field--> [domain] machine learning (gate 1.16) --inv. in field--> [paper] Recurrent World Models Facilitate Policy Evolution

**frame graph, gate off** (ranks: fused 9, graph 2477, scorer 7, dense 66)
- w=15.94: [function] characterize learning dynamics at the parameter… (gate 1.00) --inv. achieves--> [method] rlvr (gate 1.00) --inv. contributes--> [paper] The Path Not Taken: RLVR Provably Learns Off the Principals (gate 1.00) --in field--> [domain] machine learning (gate 1.16) --inv. in field--> [paper] Recurrent World Models Facilitate Policy Evolution

**OpenIE graph** (ranks: fused 16, graph 2566, scorer 14, dense 66)
- w=0.02: language agents --is mentioned in--> [paper] MLAgentBench: Evaluating Language Agents on Machine Learnin… --inv. is mentioned in--> agent --is mentioned in--> [paper] Recurrent World Models Facilitate Policy Evolution

## 4. 10.48550_arxiv.2602.04572  ->  Two-Person Cooperative Games

**Field:** Computer Science -> Decision Sciences | **stratum:** cross | **golds for this query:** 5 | **route:** bridge, 4 hops (shortest 4)

**Query:** How can a generative AI system and a Q&A forum achieve sustainable, mutually beneficial collaboration without monetary exchange, when each possesses private information about its own utilities and their incentives are intrinsically misaligned? Existing proposals for sustaining Q&A forums in the face of generative AI adoption focus on restricting data access, offering financial compensation, pricing data, or designing behavioral interventions to steer user behavior. These approaches treat AI providers and knowledge communities as adversarial competitors over data and attention, rather than as interdependent strategic agents. They fail to model forums as autonomous stakeholders with their own…

**Inspiration (gold):** Two-Person Cooperative Games. The concept of a general two-person cooperative game is defined and a concept of a solution for such games is developed.

**Ranks of this gold:** SciGraphIR 19, graph channel 372, multi-view scorer 18, Qwen3 cosine 483, frame graph, gate off|fused 20, frame graph, gate off|graph 1765, OpenIE graph|fused 12, OpenIE graph|graph 2540, Qwen3-Emb. >300

**Scorer views that matched best (hypothetical answers written for the query):**
- view 3 match 0.514: A non-parametric estimation technique for latent utility functions of both generative AI systems and forum stakeholders, enabling the discovery of hidden preferences and potential areas for cooperative engagement.

**SciGraphIR (frame graph + CCMP)** (ranks: fused 19, graph 372, scorer 18, dense 483)
- w=16.63: [function] model agents with subjective beliefs (gate 1.00) --inv. achieves--> [method] berk-nash equilibrium (gate 1.00) --inv. contributes--> [paper] Berk-Nash Equilibrium: A Framework for Modeling Agents With… (gate 1.00) --in field--> [domain] game theory (gate 1.14) --inv. in field--> [paper] Two-Person Cooperative Games

**frame graph, gate off** (ranks: fused 20, graph 1765, scorer 18, dense 483)
- w=15.75: [function] model agents with subjective beliefs (gate 1.00) --inv. achieves--> [method] berk-nash equilibrium (gate 1.00) --inv. contributes--> [paper] Berk-Nash Equilibrium: A Framework for Modeling Agents With… (gate 1.00) --in field--> [domain] game theory (gate 1.14) --inv. in field--> [paper] Two-Person Cooperative Games

**OpenIE graph** (ranks: fused 12, graph 2540, scorer 12, dense 483)
- (no valid path within the reasoner's depth)

## 5. 10.48550_arxiv.2602.01456  ->  On the dimension and entropy of probability distributions.

**Field:** Computer Science -> Physics and Astronomy | **stratum:** cross | **golds for this query:** 7 | **route:** bridge, 3 hops (shortest 3)

**Query:** How can joint-embedding predictive architectures be regularized to learn sparse, non-negative representations while still preserving maximum-entropy and preventing collapse? Prior methods for preventing collapse in joint-embedding predictive architectures regularize representations by matching them, through one-dimensional projections, to isotropic Gaussian distributions, which are maximum-entropy under an expected squared-norm constraint. This yields dense representations and fails to capture sparsity, a key property of efficient and interpretable neural codes. Existing approaches to sparsity rely on explicit penalties or architectural constraints rather than distribution matching, and non…

**Inspiration (gold):** On the dimension and entropy of probability distributions.

**Ranks of this gold:** SciGraphIR 21, graph channel 687, multi-view scorer 20, Qwen3 cosine 69, frame graph, gate off|fused 22, frame graph, gate off|graph 1675, OpenIE graph|fused 27, OpenIE graph|graph 3756, Qwen3-Emb. 69

**Scorer views that matched best (hypothetical answers written for the query):**
- view 0 match 0.598: A variational autoencoder with a Kullback-Leibler divergence term that regularizes the latent space to follow a sparse, non-negative prior distribution, ensuring maximum entropy by using a Gaussian mixture model as the prior.

**SciGraphIR (frame graph + CCMP)** (ranks: fused 21, graph 687, scorer 20, dense 69)
- w=2.30: [limitation] lack of efficient algorithms for certain distr… (gate 1.00) --inv. same limitation--> [limitation] difficulty in assessing distribution complexity (gate 1.00) --inv. overcomes--> [method] entropy dimension analysis (gate 1.00) --inv. contributes--> [paper] On the dimension and entropy of probability distributions.

**frame graph, gate off** (ranks: fused 22, graph 1675, scorer 20, dense 69)
- w=2.40: [limitation] lack of efficient algorithms for certain distr… (gate 1.00) --inv. same limitation--> [limitation] difficulty in assessing distribution complexity (gate 1.00) --inv. overcomes--> [method] entropy dimension analysis (gate 1.00) --inv. contributes--> [paper] On the dimension and entropy of probability distributions.

**OpenIE graph** (ranks: fused 27, graph 3756, scorer 27, dense 69)
- (no valid path within the reasoner's depth)

## 6. 10.48550_arxiv.2602.21315  ->  Tail bounds for sums of geometric and exponential variables

**Field:** Computer Science -> Mathematics | **stratum:** cross | **golds for this query:** 3 | **route:** bridge, 6 hops (shortest 5)

**Query:** Is there any backoff protocol that is stable for a positive arrival rate? Prior analyses of backoff protocols established instability only for specific send sequences, such as those with constant or exponentially decreasing send probabilities, or for sequences satisfying a structural 'coverage' condition that guarantees the expected number of sends remains sufficiently high. These methods relied on sustained growth of expected noise or on a characterization of protocols that almost surely have only finitely many successful sends. They could not handle arbitrary send sequences, especially those with many low-weight bins or with infinitely many 'exposed' bins that cause the expected noise to…

**Inspiration (gold):** Tail bounds for sums of geometric and exponential variables. We give explicit bounds for the tail probabilities for sums of independent geometric or exponential variables, possibly with different parameters.

**Ranks of this gold:** SciGraphIR 21, graph channel 900, multi-view scorer 19, Qwen3 cosine 78, frame graph, gate off|fused 21, frame graph, gate off|graph 1001, OpenIE graph|fused 17, OpenIE graph|graph 3808, Qwen3-Emb. 74

**Scorer views that matched best (hypothetical answers written for the query):**
- view 0 match 0.482: A stochastic model based on Markov chains to analyze the stability of backoff protocols under arbitrary send sequences, employing transition probabilities that account for both successful and failed transmissions.

**SciGraphIR (frame graph + CCMP)** (ranks: fused 21, graph 900, scorer 19, dense 78)
- w=0.06: [function] characterize the behavior of sequences of random… (gate 1.00) --is specific case of--> [function] characterize asymptotic behavior in general sett… (gate 1.00) --is specific case of--> [function] distribution behavior near convergence (gate 1.00) --inv. concerns--> [finding] improved approximation accuracy (gate 0.89) --inv. reports--> [paper] Sums of independent random variables (gate 0.82) --in field--> [domain] probability theory (gate 1.35) --inv. in field--> [paper] Tail bounds for sums of geometric and exponential variables

**frame graph, gate off** (ranks: fused 21, graph 1001, scorer 19, dense 78)
- w=0.04: [function] characterize the behavior of sequences of random… (gate 1.00) --is specific case of--> [function] characterize asymptotic behavior in general sett… (gate 1.00) --is specific case of--> [function] distribution behavior near convergence (gate 1.00) --inv. concerns--> [finding] improved approximation accuracy (gate 0.89) --inv. reports--> [paper] Sums of independent random variables (gate 0.82) --in field--> [domain] probability theory (gate 1.35) --inv. in field--> [paper] Tail bounds for sums of geometric and exponential variables

**OpenIE graph** (ranks: fused 17, graph 3808, scorer 16, dense 78)
- w=-0.00: erasure protocol --is mentioned in--> [paper] Capturing the Landauer bound through the application of a d… --inv. is mentioned in--> numerical simulations --inv. is validated by--> theory --is mentioned in--> [paper] Universality and Sharp Matrix Concentration Inequalities --inv. is mentioned in--> independent --is mentioned in--> [paper] Tail bounds for sums of geometric and exponential variables

## 7. 10.48550_arxiv.2602.13672  ->  Forecasting: principles and practice.

**Field:** Computer Science -> Decision Sciences | **stratum:** cross | **golds for this query:** 4 | **route:** bridge, 4 hops (shortest 4)

**Query:** How can intent drift, a gradual and subtle divergence of a network's operational state from its intended goal, be detected in real time to provide early warnings and enable proactive prevention of service failures while minimizing false alarms? Prior approaches to intent drift detection include unsupervised anomaly detection on network telemetry, which struggles to separate benign anomalies from critical drift; distance-based methods that measure the deviation of operational KPI vectors from target states, which only trigger alerts after substantial deviation has already occurred; and predictive models that forecast individual KPIs or classify near-term network conditions, which are indirec…

**Inspiration (gold):** Forecasting: principles and practice.

**Ranks of this gold:** SciGraphIR 13, graph channel 907, multi-view scorer 13, Qwen3 cosine 43, frame graph, gate off|fused 15, frame graph, gate off|graph 2139, OpenIE graph|fused 19, OpenIE graph|graph 3086, Qwen3-Emb. 45

**Scorer views that matched best (hypothetical answers written for the query):**
- view 0 match 0.538: A recurrent neural network (RNN) model that processes time-series data of network KPIs to predict imminent intent drift by identifying patterns indicative of deviation from target operational states, utilizing backpropagation through time…

**SciGraphIR (frame graph + CCMP)** (ranks: fused 13, graph 907, scorer 13, dense 43)
- w=17.14: [method] distance-based methods (gate 1.00) --inv. builds on--> [method] maximum likelihood estimator (gate 1.00) --inv. contributes--> [paper] Maximum Likelihood Estimation of Intrinsic Dimension (gate 1.00) --in field--> [domain] statistics (gate 1.17) --inv. in field--> [paper] Forecasting: principles and practice.

**frame graph, gate off** (ranks: fused 15, graph 2139, scorer 13, dense 43)
- w=17.64: [method] distance-based methods (gate 1.00) --inv. builds on--> [method] maximum likelihood estimator (gate 1.00) --inv. contributes--> [paper] Maximum Likelihood Estimation of Intrinsic Dimension (gate 1.00) --in field--> [domain] statistics (gate 1.17) --inv. in field--> [paper] Forecasting: principles and practice.

**OpenIE graph** (ranks: fused 19, graph 3086, scorer 18, dense 43)
- (no valid path within the reasoner's depth)

## 8. 10.48550_arxiv.2602.22831  ->  The Moral Machine experiment.

**Field:** Computer Science -> Neuroscience | **stratum:** cross | **golds for this query:** 4 | **route:** none, 0 hops (shortest 2)

**Query:** How do directed contextual influences in prompts shift moral triage decisions in large language models, and does the direction of the influence determine the size and reliability of the shift? Prior approaches evaluate LLM moral behavior by presenting context-minimal forced-choice dilemmas and measuring baseline preferences. Other approaches probe with rich adversarial or narrative contexts, but are open-ended and produce qualitative failures rather than stable aggregate statistics. The field lacked a controlled quantitative method for measuring how contextual signals alter decisions relative to baseline, and for comparing those effects across models and settings.

**Inspiration (gold):** The Moral Machine experiment.

**Ranks of this gold:** SciGraphIR 2, graph channel 5, multi-view scorer 2, Qwen3 cosine 111, frame graph, gate off|fused 2, frame graph, gate off|graph 5, OpenIE graph|fused 1, OpenIE graph|graph 1, Qwen3-Emb. 123

**Scorer views that matched best (hypothetical answers written for the query):**
- view 1 match 0.662: An experimental design utilizing counterbalanced within-subjects comparisons was implemented, where participants rated moral dilemmas presented with both neutral and contextually rich prompts, measuring decision shifts in moral triage usin…

**SciGraphIR (frame graph + CCMP)** (ranks: fused 2, graph 5, scorer 2, dense 111)
- (no valid path within the reasoner's depth)

**frame graph, gate off** (ranks: fused 2, graph 5, scorer 2, dense 111)
- (no valid path within the reasoner's depth)

**OpenIE graph** (ranks: fused 1, graph 1, scorer 3, dense 111)
- w=16.88: moral machine --is mentioned in--> [paper] The Moral Machine experiment.

## 9. 10.48550_arxiv.2601.22474  ->  The effect of the introduction of reward upon the maze performance of rats.

**Field:** Computer Science -> Neuroscience | **stratum:** cross | **golds for this query:** 5 | **route:** none, 0 hops (shortest 6)

**Query:** Can large language models exhibit latent learning, improving their performance through unrewarded exploration and then benefiting more from subsequent reward-based training? Existing approaches to fine-tune large language models for reasoning and interactive tasks predominantly use reinforcement learning driven by scalar reward signals, requiring external verification or human feedback to assign credit to generated responses. These methods treat reward as essential for learning, and do not account for the possibility that agents can acquire useful internal representations or knowledge during exploration without any reward signal, as demonstrated in classical psychological studies of latent…

**Inspiration (gold):** The effect of the introduction of reward upon the maze performance of rats.

**Ranks of this gold:** SciGraphIR 1, graph channel 1965, multi-view scorer 1, Qwen3 cosine 957, frame graph, gate off|fused 1, frame graph, gate off|graph 238, OpenIE graph|fused 1, OpenIE graph|graph 4164, Qwen3-Emb. >300

**Scorer views that matched best (hypothetical answers written for the query):**
- view 4 match 0.694: In psychology, a hypothetical study examining how rats in a maze can develop cognitive maps through exploration without rewards, which later facilitate faster navigation to food when rewards are introduced.

**SciGraphIR (frame graph + CCMP)** (ranks: fused 1, graph 1965, scorer 1, dense 957)
- (no valid path within the reasoner's depth)

**frame graph, gate off** (ranks: fused 1, graph 238, scorer 1, dense 957)
- (no valid path within the reasoner's depth)

**OpenIE graph** (ranks: fused 1, graph 4164, scorer 1, dense 957)
- (no valid path within the reasoner's depth)

## 10. 10.48550_arxiv.2601.22036  ->  Wasserstein geometry of Gaussian measures

**Field:** Computer Science -> Mathematics | **stratum:** cross | **golds for this query:** 3 | **route:** bridge, 4 hops (shortest 3)

**Query:** How can we quantify the degree of fusion or separation between two latent representation groups in a way that is sensitive to geometric displacement and within-group dispersion while remaining invariant to global scaling, structural deformation, and outliers? Existing distributional distance measures for comparing latent representations conflate multiple distinct factors into a single scalar. Optimal transport-based distances compute the minimum cost of transforming one distribution into another, jointly penalizing inter-group translation and internal structural deformation, so they cannot separate genuine separation from intrinsic shape differences. Kernel-based distances compare distribut…

**Inspiration (gold):** Wasserstein geometry of Gaussian measures. Abstract. This paper concerns the Riemannian/Alexandrov geometry of Gaussian measures, from the view point of the L 2-Wasserstein geometry. The space of Gaussian measures is of finite dimension, which allows to write down the explicit Riemannian metric which in turn induces the L 2-Wasserstein distance. Moreover, its completion as a metric space provides a complete picture of the singular behavior of the L 2-Wasserstein geometry. In particular, the sin…

**Ranks of this gold:** SciGraphIR 19, graph channel 1015, multi-view scorer 17, Qwen3 cosine 46, frame graph, gate off|fused 19, frame graph, gate off|graph 189, OpenIE graph|fused 17, OpenIE graph|graph 3569, Qwen3-Emb. 43

**Scorer views that matched best (hypothetical answers written for the query):**
- view 0 match 0.521: A Wasserstein distance calculation utilizing a multi-scale approach that decomposes the distance into components sensitive to both geometric displacement and within-group dispersion, while applying a robust scaling invariant transformation…

**SciGraphIR (frame graph + CCMP)** (ranks: fused 19, graph 1015, scorer 17, dense 46)
- w=10.41: [function] provide a measure of connectivity in a geometric… (gate 1.00) --inv. achieves--> [method] geodetic number (gate 1.00) --inv. contributes--> [paper] Geodetic Number versus Hull Number in P3-Convexity. (gate 1.00) --in field--> [domain] mathematics / geometry (gate 0.89) --inv. in field--> [paper] Wasserstein geometry of Gaussian measures

**frame graph, gate off** (ranks: fused 19, graph 189, scorer 17, dense 46)
- w=10.72: [function] provide a measure of connectivity in a geometric… (gate 1.00) --inv. achieves--> [method] geodetic number (gate 1.00) --inv. contributes--> [paper] Geodetic Number versus Hull Number in P3-Convexity. (gate 1.00) --in field--> [domain] mathematics / geometry (gate 0.89) --inv. in field--> [paper] Wasserstein geometry of Gaussian measures

**OpenIE graph** (ranks: fused 17, graph 3569, scorer 18, dense 46)
- w=0.10: hausdorff dimension --inv. has--> singular set --is mentioned in--> [paper] Wasserstein geometry of Gaussian measures

## 11. 10.48550_arxiv.2602.02193  ->  Mathematical Analysis of Singularities in the Diffusion Model Under the Submanifold Assumption

**Field:** Computer Science -> Engineering | **stratum:** cross | **golds for this query:** 3 | **route:** hub, 3 hops (shortest 1)

**Query:** Why does inverting a real image into the initial noise of a diffusion model produce non-Gaussian noise with poor editability, and how can inversion be made stable and yield editable Gaussian noise while preserving reconstruction fidelity? Existing inversion methods for diffusion models typically reverse the deterministic sampling process by numerically integrating the reverse ordinary differential equation, often relying on approximations that treat the denoiser's output as locally constant. Other approaches incorporate iterative correction steps, optimize latent embeddings to minimize reconstruction error, or design specialized numerical schemes to improve fidelity. Some methods also regul…

**Inspiration (gold):** Mathematical Analysis of Singularities in the Diffusion Model Under the Submanifold Assumption. This paper concerns the mathematical analyses of the diffusion model in machine learning. The drift term of the backward sampling process is represented as a conditional expectation involving the data distribution and the forward diffusion. The training process aims to find such a drift function by minimizing the mean-squared residue related to the conditional expectation. Using small-time approximat…

**Ranks of this gold:** SciGraphIR 23, graph channel 1, multi-view scorer 80, Qwen3 cosine 81, frame graph, gate off|fused 20, frame graph, gate off|graph 1, Qwen3-Emb. 81

**Scorer views that matched best (hypothetical answers written for the query):**
- view 0 match 0.510: A variational inference approach utilizing a Gaussian mixture model to estimate the posterior distribution of latent variables, allowing for the incorporation of non-Gaussian noise characteristics while preserving the underlying structure…

**SciGraphIR (frame graph + CCMP)** (ranks: fused 23, graph 1, scorer 80, dense 81)
- w=13.21: [task] diffusion model optimization (gate 1.00) --inv. addresses--> [paper] DilateQuant: Accurate and Efficient Diffusion Quantization… (gate 1.00) --in field--> [domain] machine learning (gate 1.00) --inv. in field--> [paper] Mathematical Analysis of Singularities in the Diffusion Mod…

**frame graph, gate off** (ranks: fused 20, graph 1, scorer 80, dense 81)
- w=18.25: [task] mathematical analysis of diffusion models (gate 1.00) --inv. addresses--> [paper] Mathematical Analysis of Singularities in the Diffusion Mod…

**OpenIE graph**: no interpretation

## 12. 10.48550_arxiv.2602.03305  ->  Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping

**Field:** Computer Science -> Economics, Econometrics and Finance | **stratum:** cross | **golds for this query:** 4 | **route:** hub, 3 hops (shortest 2)

**Query:** How can we systematically design and verify reward functions for offline clinical reinforcement learning that are safe, effective, and generalizable across diverse diseases? Prior reward designs for clinical RL use either sparse terminal outcomes, dense hand-crafted intermediate rewards, or a combination, but they rely on manual feature selection and ad-hoc parameter tuning that do not transfer across diseases. These methods suffer from sparse credit assignment, misaligned objectives such as reward hacking, and no way to verify the reward signal offline. Directly using large language models to score trajectories or write reward code also fails because they lack temporal credit assignment an…

**Inspiration (gold):** Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. This paper investigates conditions under which modifications to the reward function of a Markov decision process preserve the optimal policy. It is shown that, besides the positive linear transformation familiar from utility theory, one can add a reward for transitions between states that is expressible as the difference in value of an arbitrary potential function applied to those states. Furthermore, this…

**Ranks of this gold:** SciGraphIR 11, graph channel 4, multi-view scorer 11, Qwen3 cosine 29, frame graph, gate off|fused 11, frame graph, gate off|graph 1, OpenIE graph|fused 13, OpenIE graph|graph 1707, Qwen3-Emb. 30

**Scorer views that matched best (hypothetical answers written for the query):**
- view 2 match 0.640: An adversarial training approach where a discriminator is trained to differentiate between safe and unsafe reward functions, providing a mechanism to verify the robustness of the reward design in offline settings.

**SciGraphIR (frame graph + CCMP)** (ranks: fused 11, graph 4, scorer 11, dense 29)
- w=14.16: [task] reward design for reinforcement learning (gate 1.00) --inv. addresses--> [paper] A large language model-driven reward design framework via d… (gate 1.00) --in field--> [domain] reinforcement learning (gate 1.00) --inv. in field--> [paper] Policy Invariance Under Reward Transformations: Theory and…

**frame graph, gate off** (ranks: fused 11, graph 1, scorer 11, dense 29)
- w=19.50: [task] reward design for reinforcement learning (gate 1.00) --inv. addresses--> [paper] A large language model-driven reward design framework via d… (gate 1.00) --in field--> [domain] reinforcement learning (gate 1.00) --inv. in field--> [paper] Policy Invariance Under Reward Transformations: Theory and…

**OpenIE graph** (ranks: fused 13, graph 1707, scorer 11, dense 29)
- w=0.06: reward functions --equivalent--> reward function --is mentioned in--> [paper] Policy Invariance Under Reward Transformations: Theory and…

## 13. 10.1111_coin.70150  ->  Statistical methodologies to pool across multiple intervention studies

**Field:** Computer Science -> Mathematics | **stratum:** cross | **golds for this query:** 4 | **route:** none, 0 hops (shortest 5)

**Query:** How can the aggregation of locally trained models in federated learning be made more robust to the heterogeneity of data distributions across clients? Existing methods for combining locally trained models in distributed learning typically compute a weighted average of model parameters, with weights proportional to the number of local training samples. This averaging scheme assumes that all clients provide equally reliable estimates, but when data is non-identically distributed across clients, some models may be highly uncertain or noisy. This uncertainty is not reflected in the aggregation, leading to degraded performance and slower convergence. Other strategies, such as sharing a small sub…

**Inspiration (gold):** Statistical methodologies to pool across multiple intervention studies. Combining and analyzing data from heterogeneous randomized controlled trials of complex multiple-component intervention studies, or discussing them in a systematic review, is not straightforward. The present article describes certain issues to be considered when combining data across studies, based on discussions in an NIH-sponsored workshop on pooling issues across studies in consortia (see Belle et al. in Psychol Aging, 1…

**Ranks of this gold:** SciGraphIR 9, graph channel 1144, multi-view scorer 9, Qwen3 cosine 67, frame graph, gate off|fused 9, frame graph, gate off|graph 58, OpenIE graph|fused 15, OpenIE graph|graph 2776, Qwen3-Emb. 65

**Scorer views that matched best (hypothetical answers written for the query):**
- view 6 match 0.500: In the field of finance, a paper proposes a method to combine forecasts from various economic indicators using robust statistical techniques that account for the heterogeneity in the data sources, improving overall predictive performance.

**SciGraphIR (frame graph + CCMP)** (ranks: fused 9, graph 1144, scorer 9, dense 67)
- (no valid path within the reasoner's depth)

**frame graph, gate off** (ranks: fused 9, graph 58, scorer 9, dense 67)
- (no valid path within the reasoner's depth)

**OpenIE graph** (ranks: fused 15, graph 2776, scorer 13, dense 67)
- w=0.00: multi task learning --inv. benefits from--> deep learning --is a type of--> ai --is mentioned in--> [paper] Thinking Assistants: LLM-Based Conversational Assistants th… --inv. is mentioned in--> participants --is mentioned in--> [paper] Statistical methodologies to pool across multiple intervent…

## 14. 10.48550_arxiv.2602.02431  ->  Gradient Descent Only Converges to Minimizers: Non-Isolated Critical Points and Invariant Regions

**Field:** Computer Science -> Medicine | **stratum:** cross | **golds for this query:** 8 | **route:** other, 1 hops (shortest 1)

**Query:** Can full-batch gradient descent, by reusing the training data across iterations, overcome the extra sqrt(d) factor in sample complexity that one-pass SGD suffers when learning an even single-index target function, and if so, what iteration complexity is required to achieve weak and strong recovery? Prior work on gradient-based learning of single-index models focused on one-pass (online) updates, showing that for link functions whose low-degree expansion starts at quadratic order, weak recovery requires on the order of d^{3/2} samples when the learning rate is capped for numerical stability. Analyses of full-batch (multi-pass) gradient methods either required sample sizes with polylogarithmi…

**Inspiration (gold):** Gradient Descent Only Converges to Minimizers: Non-Isolated Critical Points and Invariant Regions. Given a twice continuously differentiable cost function f, we prove that the set of initial conditions so that gradient descent converges to saddle points where \nabla^2 f has at least one strictly negative eigenvalue, has (Lebesgue) measure zero, even for cost functions f with non-isolated critical points, answering an open question in [Lee, Simchowitz, Jordan, Recht, COLT 2016]. Moreover, this r…

**Ranks of this gold:** SciGraphIR 11, graph channel 2, multi-view scorer 25, Qwen3 cosine 28, frame graph, gate off|fused 10, frame graph, gate off|graph 2, Qwen3-Emb. 28

**Scorer views that matched best (hypothetical answers written for the query):**
- view 0 match 0.653: A full-batch gradient descent algorithm employing a modified learning rate schedule, where the learning rate is adjusted based on the cumulative gradient updates across multiple iterations, thereby allowing for convergence guarantees under…

**SciGraphIR (frame graph + CCMP)** (ranks: fused 11, graph 2, scorer 25, dense 28)
- w=11.19: [task] gradient descent convergence analysis (gate 1.00) --inv. addresses--> [paper] Gradient Descent Only Converges to Minimizers: Non-Isolated…

**frame graph, gate off** (ranks: fused 10, graph 2, scorer 25, dense 28)
- w=13.50: [task] gradient descent convergence analysis (gate 1.00) --inv. addresses--> [paper] Gradient Descent Only Converges to Minimizers: Non-Isolated…

**OpenIE graph**: no interpretation

## 15. 10.48550_arxiv.2602.16990  ->  Conditional logit analysis of qualitative choice behavior.

**Field:** Computer Science -> Social Sciences | **stratum:** cross | **golds for this query:** 7 | **route:** none, 0 hops (shortest 4)

**Query:** How can a conversational and longitudinal benchmark for stock recommendation be designed to evaluate LLMs beyond behavioral imitation, grounding assessment in investor-specific utility and enabling diagnosis of whether models follow rational analysis, mimic user noise, or are driven by market momentum? Existing recommendation benchmarks evaluate personalization primarily by behavioral imitation, using a single relevance signal such as clicks or ratings, and thus treat observed user behavior as the sole ground truth. Consumer-domain benchmarks with conversational interaction still rely on such relevance signals and lack utility grounding. Financial datasets, meanwhile, focus on prediction or…

**Inspiration (gold):** Conditional logit analysis of qualitative choice behavior.

**Ranks of this gold:** SciGraphIR 17, graph channel 1196, multi-view scorer 15, Qwen3 cosine 125, frame graph, gate off|fused 16, frame graph, gate off|graph 113, Qwen3-Emb. 127

**Scorer views that matched best (hypothetical answers written for the query):**
- view 0 match 0.682: A longitudinal mixed-methods design using a structural equation model to estimate the relationships between user-specific risk preferences, model recommendations, and decision quality over time, employing maximum likelihood estimation to a…

**SciGraphIR (frame graph + CCMP)** (ranks: fused 17, graph 1196, scorer 15, dense 125)
- (no valid path within the reasoner's depth)

**frame graph, gate off** (ranks: fused 16, graph 113, scorer 15, dense 125)
- (no valid path within the reasoner's depth)

**OpenIE graph**: no interpretation

