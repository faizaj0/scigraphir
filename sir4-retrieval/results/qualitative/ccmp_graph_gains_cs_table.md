# SIR-4 CS cross queries: CCMP gate on vs off on the SAME weights, graph-channel rank. Kept: SciGraphIR fused <=25, cosine >=30, gate-off graph rank >= 3x gate-on and >=50 ranks worse.

11 golds (of 493 cross golds with paths).

| # | graph on / off | fused on / off | cosine | scorer | OpenIE model | route (top path) | max gate | QUARTET | gold | query |
|--:|--:|--:|--:|--:|--:|---|--:|---|---|---|
| 1 | **184** / 2477 | 6 / 9 | 66 | 7 | 16 | valid, 4 hops | 1.16 | Computer Science -> Social Sciences / cross | Recurrent World Models Facilitate Policy Evolution | What are the capabilities of current language-model-based agents in long-horizon |
| 2 | **279** / 2192 | 16 / 16 | 87 | 13 | 13 | valid, 4 hops | 1.16 | Computer Science -> Computer Science / same | Chain-Of-Thought Prompting Elicits Reasoning in Large Langua | How can we make LLM-driven theorem proving practical for real-world industrial-s |
| 3 | **67** / 1781 | 23 / 24 | 56 | 21 | 24 | valid, 3 hops | 1.00 | Computer Science -> Computer Science / same | Alphazero-like Tree-Search can Guide Large Language Model De | How can LLM-based table reasoning achieve reliable step-level verification, effe |
| 4 | **147** / 1837 | 18 / 21 | 71 | 19 | 29 | valid, 4 hops | 1.18 | None / None | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via  | How can vision-language models be trained to perform robust spatial reasoning an |
| 5 | **41** / 1711 | 22 / 27 | 30 | 25 | None | valid, 4 hops | 1.18 | Computer Science -> Social Sciences / same | Do Large Language Models Perform the Way People Expect? Meas | What characteristics of an algorithm determine whether users can form accurate p |
| 6 | **45** / 1699 | 8 / 8 | 260 | 6 | 6 | valid, 3 hops | 1.00 | Computer Science -> Computer Science / same | Focal Loss for Dense Object Detection | How can we fine-tune a pre-trained deep learning model on a small, imbalanced da |
| 7 | **22** / 1648 | 24 / 26 | 167 | 25 | 30 | valid, 3 hops | 1.00 | Computer Science -> Decision Sciences / same | Preference-based Online Learning with Dueling Bandits: A Sur | How can we optimize candidates in a large discrete output space for test-time LL |
| 8 | **33** / 1641 | 2 / 3 | 2357 | 2 | 2 | valid, 4 hops | 1.16 | Computer Science -> Computer Science / same | Attention Is All You Need | How can we accurately predict the total widths of mesons from their quantum numb |
| 9 | **27** / 1423 | 18 / 19 | 149 | 17 | 17 | valid, 4 hops | 1.14 | Computer Science -> Computer Science / same | A simple framework for contrastive learning of visual repres | How can an agent learn a world model from pixel observations that extrapolates t |
| 10 | **372** / 1765 | 19 / 20 | 483 | 18 | 12 | valid, 4 hops | 1.14 | Computer Science -> Decision Sciences / cross | Two-Person Cooperative Games | How can a generative AI system and a Q&A forum achieve sustainable, mutually ben |
| 11 | **602** / 1894 | 6 / 6 | 68 | 5 | 5 | valid, 4 hops | 1.18 | Computer Science -> Computer Science / same | BERT: Pre-training of Deep Bidirectional Transformers for La | Can a single unified encoder trained jointly on audio, visual, and text modaliti |

full dump of all 11 -> results/qualitative/inspect_cs_ccmp_graph_gains.txt
