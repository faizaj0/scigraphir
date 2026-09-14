# Why the walk prefers one graph: seeds, distances, floods

| dataset | graph | stratum | seeds/query | median seed degree | hub seeds (deg>200) | gold at d=1 | d=2 | d=3 | d>3 | docs within 2 hops (median) | walk nDCG@5 all | d=1 queries | d=2 queries | d>=3 queries |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| sir4_cs | SciAffordGraph | all | 38.7 | 6 | 0.0% | 18.2% | 24.3% | 29.2% | 28.4% | 641 | 19.49 | 38.3 (n=511) | 2.9 (n=318) | 0.3 (n=174) |
| sir4_cs | SciAffordGraph | same | 38.5 | 6 | 0.0% | 19.6% | 25.1% | 29.3% | 26.0% | 602 | 20.24 | 38.8 (n=403) | 2.6 (n=240) | 0.2 (n=131) |
| sir4_cs | SciAffordGraph | cross | 39.0 | 6 | 0.0% | 13.9% | 21.8% | 28.8% | 35.4% | 712 | 17.05 | 36.1 (n=108) | 3.8 (n=78) | 0.5 (n=43) |
| sir4_cs | OpenIE | all | 9.2 | 3 | 2.2% | 26.4% | 19.2% | 25.6% | 28.8% | 1544 | 13.25 | 20.5 (n=673) | 0.1 (n=189) | 0.0 (n=128) |
| sir4_cs | OpenIE | same | 9.3 | 3 | 2.0% | 27.0% | 19.9% | 26.2% | 26.8% | 1519 | 13.47 | 20.5 (n=522) | 0.0 (n=144) | 0.0 (n=90) |
| sir4_cs | OpenIE | cross | 8.7 | 4 | 3.0% | 24.4% | 17.3% | 23.8% | 34.5% | 1623 | 12.50 | 20.3 (n=151) | 0.3 (n=45) | 0.0 (n=38) |
| sir4_biology | SciAffordGraph | all | 32.0 | 5 | 0.0% | 20.2% | 26.2% | 26.7% | 26.9% | 413 | 25.59 | 44.4 (n=441) | 5.9 (n=257) | 0.4 (n=106) |
| sir4_biology | SciAffordGraph | same | 31.7 | 5 | 0.0% | 21.2% | 25.8% | 27.2% | 25.7% | 414 | 26.36 | 45.9 (n=370) | 5.2 (n=207) | 0.3 (n=89) |
| sir4_biology | SciAffordGraph | cross | 33.4 | 5 | 0.0% | 15.3% | 28.1% | 24.3% | 32.2% | 398 | 21.82 | 36.7 (n=71) | 8.9 (n=50) | 1.2 (n=17) |
| sir4_biology | OpenIE | all | 7.1 | 4 | 0.2% | 45.2% | 14.3% | 17.9% | 22.6% | 601 | 28.84 | 33.9 (n=699) | 0.8 (n=54) | 0.0 (n=38) |
| sir4_biology | OpenIE | same | 7.0 | 4 | 0.1% | 46.5% | 14.3% | 18.0% | 21.2% | 560 | 29.59 | 34.3 (n=587) | 0.3 (n=38) | 0.0 (n=31) |
| sir4_biology | OpenIE | cross | 7.5 | 4 | 0.2% | 39.3% | 14.5% | 17.3% | 28.9% | 785 | 25.19 | 31.4 (n=112) | 1.9 (n=16) | 0.0 (n=7) |
| sir4_physics | SciAffordGraph | all | 31.8 | 6 | 0.0% | 21.7% | 28.4% | 33.0% | 16.9% | 512 | 24.73 | 44.0 (n=443) | 4.2 (n=263) | 0.2 (n=103) |
| sir4_physics | SciAffordGraph | same | 31.6 | 6 | 0.0% | 23.0% | 28.9% | 33.9% | 14.3% | 510 | 25.86 | 44.2 (n=386) | 4.5 (n=206) | 0.2 (n=85) |
| sir4_physics | SciAffordGraph | cross | 32.9 | 5 | 0.0% | 15.5% | 26.5% | 28.8% | 29.2% | 544 | 19.02 | 43.1 (n=57) | 2.9 (n=57) | 0.0 (n=18) |
| sir4_physics | OpenIE | all | 8.0 | 4 | 0.0% | 41.4% | 17.4% | 18.3% | 22.8% | 562 | 22.86 | 29.2 (n=651) | 0.4 (n=83) | 0.4 (n=53) |
| sir4_physics | OpenIE | same | 8.0 | 4 | 0.0% | 43.5% | 18.0% | 18.7% | 19.8% | 538 | 23.45 | 29.4 (n=553) | 0.3 (n=68) | 0.5 (n=44) |
| sir4_physics | OpenIE | cross | 8.3 | 4 | 0.0% | 31.6% | 14.7% | 16.8% | 36.9% | 648 | 19.92 | 27.9 (n=98) | 0.9 (n=15) | 0.0 (n=9) |
| sir4_matsci | SciAffordGraph | all | 32.0 | 6 | 0.0% | 19.4% | 30.2% | 31.7% | 18.7% | 388 | 23.22 | 43.8 (n=167) | 3.2 (n=117) | 0.0 (n=37) |
| sir4_matsci | SciAffordGraph | same | 31.4 | 6 | 0.0% | 21.2% | 29.2% | 32.5% | 17.1% | 381 | 24.66 | 43.6 (n=114) | 3.1 (n=68) | 0.0 (n=22) |
| sir4_matsci | SciAffordGraph | cross | 33.0 | 6 | 0.0% | 16.3% | 31.9% | 30.3% | 21.5% | 397 | 20.71 | 44.3 (n=53) | 3.2 (n=49) | 0.0 (n=15) |
| sir4_matsci | OpenIE | all | 8.4 | 4 | 0.0% | 37.8% | 16.7% | 17.6% | 27.9% | 363 | 21.79 | 28.4 (n=254) | 0.0 (n=38) | 0.0 (n=20) |
| sir4_matsci | OpenIE | same | 8.3 | 4 | 0.0% | 39.3% | 18.8% | 18.1% | 23.8% | 364 | 22.69 | 29.1 (n=164) | 0.0 (n=25) | 0.0 (n=12) |
| sir4_matsci | OpenIE | cross | 8.6 | 4 | 0.0% | 35.3% | 13.1% | 16.7% | 34.9% | 361 | 20.22 | 27.2 (n=90) | 0.0 (n=13) | 0.0 (n=8) |
| tomato | SciAffordGraph | all | 72.4 | 5 | 0.0% | 28.8% | 28.0% | 27.6% | 15.6% | 107 | 14.42 | 48.7 (n=901) | 1.5 (n=877) | 0.0 (n=865) |
| tomato | SciAffordGraph | same | 72.1 | 5 | 0.0% | 29.7% | 28.4% | 27.8% | 14.1% | 105 | 14.84 | 48.6 (n=843) | 1.5 (n=807) | 0.0 (n=791) |
| tomato | SciAffordGraph | cross | 74.9 | 5 | 0.0% | 20.1% | 24.2% | 25.6% | 30.1% | 1143 | 10.35 | 50.2 (n=58) | 1.1 (n=70) | 0.0 (n=74) |
| tomato | OpenIE | all | 11.7 | 5 | 1.6% | 43.4% | 17.0% | 21.6% | 17.9% | 396 | 13.80 | 31.7 (n=1360) | 0.1 (n=534) | 0.0 (n=677) |
| tomato | OpenIE | same | 11.7 | 5 | 1.7% | 45.4% | 17.9% | 21.0% | 15.7% | 321 | 14.25 | 31.4 (n=1290) | 0.1 (n=510) | 0.0 (n=597) |
| tomato | OpenIE | cross | 11.5 | 4 | 1.4% | 24.2% | 8.3% | 27.7% | 39.8% | 1260 | 9.33 | 38.5 (n=70) | 0.0 (n=24) | 0.0 (n=80) |
| mir | SciAffordGraph | all | 22.2 | 6 | 0.0% | 26.4% | 28.1% | 32.3% | 13.2% | 638 | 12.93 | 37.2 (n=52) | 1.4 (n=48) | 0.0 (n=45) |
| mir | OpenIE | all | 4.5 | 11 | 9.8% | 35.3% | 27.7% | 23.8% | 13.2% | 1767 | 13.84 | 28.4 (n=70) | 2.4 (n=39) | 1.8 (n=36) |

seeds = the query's start nodes in that graph (frames extracted from the query on SciAffordGraph, entities linked from the query text on OpenIE). degree = undirected degree in the stage-1 graph; a hub seed spreads its mass over hundreds of neighbours. gold at d=k = share of golds whose shortest route from ANY seed is k hops (BFS, undirected). docs within 2 hops = the papers a 2-step walk can reach at all, i.e. the field the gold competes in. The last three columns are the walk's nDCG@5 on queries whose nearest gold is at that distance.

## Which seed types touch a gold directly (SciAffordGraph, distance-1 links)

| dataset | stratum | seed types per query (mean) | share of d=1 golds reached via task / function / limitation / method / entity / other |
|---|---|---|---|
| sir4_cs | all | task 3.0, function 8.0, limitation 19.6, method 8.0 | 80% / 0% / 0% / 20% / 0% / 0% |
| sir4_cs | same | task 3.0, function 8.0, limitation 19.6, method 7.9 | 80% / 0% / 0% / 20% / 0% / 0% |
| sir4_cs | cross | task 3.0, function 8.0, limitation 19.9, method 8.2 | 78% / 0% / 0% / 22% / 0% / 0% |
| sir4_biology | all | task 3.0, function 7.8, limitation 15.1, method 6.1 | 77% / 0% / 0% / 23% / 0% / 0% |
| sir4_biology | same | task 3.0, function 7.8, limitation 14.9, method 6.0 | 77% / 0% / 0% / 23% / 0% / 0% |
| sir4_biology | cross | task 3.0, function 7.8, limitation 16.1, method 6.4 | 76% / 0% / 0% / 24% / 0% / 0% |
| sir4_physics | all | task 3.0, function 7.7, limitation 15.0, method 6.0 | 76% / 0% / 0% / 24% / 0% / 0% |
| sir4_physics | same | task 3.0, function 7.7, limitation 14.9, method 5.9 | 77% / 0% / 0% / 23% / 0% / 0% |
| sir4_physics | cross | task 3.0, function 7.8, limitation 15.9, method 6.2 | 74% / 0% / 0% / 26% / 0% / 0% |
| sir4_matsci | all | task 3.0, function 7.8, limitation 15.1, method 6.1 | 78% / 0% / 0% / 22% / 0% / 0% |
| sir4_matsci | same | task 3.1, function 7.8, limitation 14.6, method 6.0 | 80% / 0% / 0% / 20% / 0% / 0% |
| sir4_matsci | cross | task 3.0, function 7.8, limitation 15.9, method 6.3 | 73% / 0% / 0% / 27% / 0% / 0% |
| tomato | all | task 8.9, function 13.1, limitation 27.4, method 17.4 | 65% / 0% / 0% / 23% / 0% / 12% |
| tomato | same | task 9.0, function 13.2, limitation 27.2, method 17.2 | 66% / 0% / 0% / 23% / 0% / 12% |
| tomato | cross | task 8.1, function 12.9, limitation 30.2, method 19.0 | 64% / 0% / 0% / 24% / 0% / 12% |
| mir | all | task 3.1, function 6.9, limitation 9.2, method 3.0 | 88% / 0% / 0% / 12% / 0% / 0% |

## Qualitative: where one graph's walk finds the gold and the other buries it

### sir4_cs: counts

frame top-5 while OpenIE past 50: 67 queries; OpenIE top-5 while frame past 50: 31 queries (cross-field: 13 vs 8)

### sir4_cs: frame graph finds it, OpenIE buries it

**10.1016_j.disc.2025.114972** (same-field) | walk rank of the gold: SciAffordGraph 1, OpenIE 1741
- query: Is the d-distance independent domination number of every tree of order n at most n/(d+1), and which trees attain equality? For trees, a classical bound gives a d-distance dominating set of size at most n/(d+1), but the constructed dominating set is not necessarily independent. For the independent ve
- gold: A note on connected bipartite graphs having independent domination number half their order.
- SciAffordGraph seeds: 22 (degrees median 4); route: [task] independent domination in bipartite graphs [deg 3] --inv. addresses--> [paper] A note on connected bipartite graphs having independent domination num [deg 4]
- OpenIE seeds: 6 (degrees median 4); route: no route within 4 hops

**10.48550_arxiv.2602.02564** (same-field) | walk rank of the gold: SciAffordGraph 5, OpenIE 1730
- query: How can we perform large-scale, multi-modal data annotation using multiple off-the-shelf AI agents without ground truth, while explicitly modeling each agent's reliability to aggregate their noisy predictions into high-confidence labels? Prior automated annotation methods treat multiple AI annotator
- gold: The Multidimensional Wisdom of Crowds. Distributing labeling tasks among hundreds or thousands of annotators is an increasingly important method for annotating large datasets. We present a method for estimating the under
- SciAffordGraph seeds: 39 (degrees median 6); route: [limitation] annotator noise can lead to inaccurate labels [deg 5] --inv. limited_by--> [task] image annotation [deg 3] --inv. addresses--> [paper] The Multidimensional Wisdom of Crowds [deg 4]
- OpenIE seeds: 7 (degrees median 3); route: annotation automation [deg 3] --equivalent--> automatic annotation pipeline [deg 6] --is_mentioned_in--> [paper] Auto4D: Learning to Label 4D Objects from Sequential Point Clouds [deg 20] --inv. is_mentioned_in--> method [deg 214] --is_mentioned_in--> [paper] The Multidimensional Wisdom of Crowds [deg 22]

**10.4204_eptcs.441.10** (same-field) | walk rank of the gold: SciAffordGraph 4, OpenIE 1347
- query: What is the logical complexity of provability in an infinite-descent proof system for first-order logic with inductively defined predicates? Existing analyses of provability complexity for inductive definition proof systems either handle only finite or regular proof trees, yielding recursively enume
- gold: The complexity of the modal predicate logic of “true in every transitive model of ZF”. Robert Solovay [8] investigated the version of the modal sentential calculus one gets by taking “□ ϕ ” to mean “ ϕ is true in every t
- SciAffordGraph seeds: 35 (degrees median 5); route: [task] relation between provability and modal logic [deg 6] --shares_purpose--> [task] axiomatization of modal logic [deg 9] --inv. addresses--> [paper] The complexity of the modal predicate logic of “true in every transiti [deg 5]
- OpenIE seeds: 20 (degrees median 3); route: proof system [deg 2] --is_mentioned_in--> [paper] A Logic of Knowing How [deg 17] --inv. is_mentioned_in--> modal logic [deg 20] --equivalent--> modal predicate calculus [deg 6] --is_mentioned_in--> [paper] The complexity of the modal predicate logic of “true in every transiti [deg 15]

### sir4_cs: OpenIE finds it, frame graph buries it

**10.48550_arxiv.2512.21507** (same-field) | walk rank of the gold: SciAffordGraph 775, OpenIE 4
- query: Can current text-to-video generation models produce socially coherent behavior—actions that reflect agents' intentions, beliefs, emotions, and social norms—rather than only physically plausible scenes? Existing evaluation benchmarks for video generation measure perceptual fidelity, temporal stabilit
- gold: Does the chimpanzee have a theory of mind?.
- SciAffordGraph seeds: 36 (degrees median 6); route: no route within 4 hops
- OpenIE seeds: 7 (degrees median 6); route: theory of mind [deg 4] --is_mentioned_in--> [paper] Does the chimpanzee have a theory of mind?. [deg 2]

**10.48550_arxiv.2602.00159** (cross-field) | walk rank of the gold: SciAffordGraph 699, OpenIE 6
- query: Can enriching graph neural networks with a learned algebraic structure--specifically, a sheaf that assigns vector spaces and linear maps to nodes and edges--improve classification accuracy on a biomedical graph dataset compared to standard graph neural networks? Standard graph neural networks propag
- gold: Toward a spectral theory of cellular sheaves. This paper outlines a program in what one might call spectral sheaf theory—an extension of spectral graph theory to cellular sheaves. By lifting the combinatorial graph Lapla
- SciAffordGraph seeds: 28 (degrees median 6); route: no route within 4 hops
- OpenIE seeds: 4 (degrees median 4); route: graph laplacian [deg 4] --equivalent--> combinatorial graph laplacian [deg 3] --is_mentioned_in--> [paper] Toward a spectral theory of cellular sheaves [deg 18]

**10.48550_arxiv.2602.22442** (same-field) | walk rank of the gold: SciAffordGraph 673, OpenIE 7
- query: How can we systematically evaluate the quality of intermediate decisions made by agent-based AutoML systems, independent of final task performance, to expose failure modes that outcome-only metrics miss? Prior agentic AutoML systems evaluate success almost exclusively through end-task metrics such a
- gold: I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo Tree Search. Recent advancements in large language models (LLMs) have shown remarkable potential in automating machine learning tasks.However, existing LLM-b
- SciAffordGraph seeds: 53 (degrees median 5); route: [limitation] lack of model interpretability [deg 18] --inv. limited_by--> [task] interpretable machine learning [deg 8] --shares_purpose--> [task] automated machine learning [deg 11] --inv. addresses--> [paper] I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo Tree Se [deg 3]
- OpenIE seeds: 3 (degrees median 6); route: f1 [deg 7] --is_mentioned_in--> [paper] Memory OS of AI Agent [deg 25] --inv. is_mentioned_in--> llms [deg 799] --is_mentioned_in--> [paper] I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo Tree Se [deg 13]

### sir4_biology: counts

frame top-5 while OpenIE past 50: 32 queries; OpenIE top-5 while frame past 50: 55 queries (cross-field: 7 vs 4)

### sir4_biology: frame graph finds it, OpenIE buries it

**10.1038_s41598-025-33445-1** (same-field) | walk rank of the gold: SciAffordGraph 2, OpenIE 1123
- query: How can we design an enhanced in silico pipeline that systematically identifies high-potential protein antigens for subunit vaccines against multidrug-resistant Staphylococcus aureus, by integrating genomic diversity, updated bioinformatics tools, and consideration of both innate and adaptive immune
- gold: AllerTOP v.2—a server for in silico prediction of allergens.
- SciAffordGraph seeds: 36 (degrees median 5); route: [function] predict allergenic potential of proteins [deg 4] --inv. achieves--> [method] allertop v.2 [deg 9] --inv. contributes--> [paper] AllerTOP v.2—a server for in silico prediction of allergens. [deg 3]
- OpenIE seeds: 4 (degrees median 10); route: no route within 4 hops

**10.3389_fpls.2025.1748099** (same-field) | walk rank of the gold: SciAffordGraph 1, OpenIE 1057
- query: How can we design allele-specific primers for simple and cost-effective PCR-based SNP detection that reliably discriminate between alleles without the need for specialized equipment? Existing SNP detection methods include high-throughput array-based platforms and sequencing approaches that are expen
- gold: SNP identification in crop plants.
- SciAffordGraph seeds: 53 (degrees median 5); route: [task] snp identification [deg 5] --inv. addresses--> [paper] SNP identification in crop plants. [deg 2]
- OpenIE seeds: 8 (degrees median 4); route: no route within 4 hops

**10.1128_mra.00118-26** (same-field) | walk rank of the gold: SciAffordGraph 1, OpenIE 881
- query: What is the taxonomic classification and genomic composition of the Alicyclobacillus strains CIJ1 and CIJ2 isolated from the Chinoike-Jigoku hot spring? Traditional taxonomic identification of Alicyclobacillus strains has relied on phenotypic traits and 16S rRNA gene phylogeny, but these methods may
- gold: TYGS and LPSN: a database tandem for fast and reliable genome-based classification and nomenclature of prokaryotes. Microbial systematics is heavily influenced by genome-based methods and challenged by an ever increasing
- SciAffordGraph seeds: 32 (degrees median 7); route: [task] genome-based classification and nomenclature of prokaryotes [deg 5] --inv. addresses--> [paper] TYGS and LPSN: a database tandem for fast and reliable genome-based cl [deg 4]
- OpenIE seeds: 5 (degrees median 4); route: no route within 4 hops

### sir4_biology: OpenIE finds it, frame graph buries it

**10.3390_vetsci12121209** (same-field) | walk rank of the gold: SciAffordGraph 700, OpenIE 969
- query: How are the expression levels of DAXX and ATRX altered in canine prostate and bladder carcinomas compared to non-malignant tissues, and do these alterations correlate with tumor aggressiveness? In human oncology, immunohistochemical assessment of DAXX and ATRX expression has been used to investigate
- gold: Overexpression of the chromatin remodeler death-domain–associated protein in prostate cancer is an independent predictor of early prostate-specific antigen recurrence.
- SciAffordGraph seeds: 34 (degrees median 6); route: [method] previous cancer biomarker studies [deg 12] --inv. improves_on--> [method] kif20a expression analysis [deg 13] --inv. contributes--> [paper] Aberrant KIF20A Expression Is Associated with Adverse Clinical Outcome [deg 5] --in_field--> [domain] oncology [deg 130] --inv. in_field--> [paper] Overexpression of the chromatin remodeler death-domain–associated prot [deg 3]
- OpenIE seeds: 10 (degrees median 4); route: canine malignant melanoma [deg 7] --is a type of--> cancer [deg 218] --is_mentioned_in--> [paper] A random walk-based method to identify driver genes by integrating the [deg 18] --inv. is_mentioned_in--> prostate cancer [deg 43] --is_mentioned_in--> [paper] Overexpression of the chromatin remodeler death-domain–associated prot [deg 3]

**10.3389_fimmu.2026.1755963** (same-field) | walk rank of the gold: SciAffordGraph 586, OpenIE 12
- query: What underlying genetic or immunological factor predisposes a previously healthy child to develop a Pseudomonas aeruginosa liver abscess? Prior approaches to investigating unusual infections in children typically involve a comprehensive immunological evaluation, including assessments of immune cell 
- gold: α1-antitrypsin promotes SPLUNC1-mediated lung defense against Pseudomonas aeruginosa infection in mice. BACKGROUND: Pseudomonas aeruginosa (PA) infection is involved in various lung diseases such as cystic fibrosis and c
- SciAffordGraph seeds: 29 (degrees median 5); route: no route within 4 hops
- OpenIE seeds: 2 (degrees median 54); route: pseudomonas aeruginosa [deg 106] --is_mentioned_in--> [paper] α1-antitrypsin promotes SPLUNC1-mediated lung defense against Pseudomo [deg 22]

**10.1186_s12920-025-02300-7** (same-field) | walk rank of the gold: SciAffordGraph 541, OpenIE 193
- query: What is the molecular impact of a novel homozygous splice site variant in the EDAR gene in a patient with hypohidrotic ectodermal dysplasia? Prior methods for characterizing disease-causing mutations relied on genetic sequencing to identify variants, but they often lacked detailed structural and fun
- gold: Molecular Dynamics Simulation for All. The impact of molecular dynamics (MD) simulations in molecular biology and drug discovery has expanded dramatically in recent years. These simulations capture the behavior of protei
- SciAffordGraph seeds: 36 (degrees median 8); route: [task] alternative splicing analysis [deg 16] --inv. addresses--> [paper] Minor intron splicing revisited: identification of new minor intron-co [deg 5] --in_field--> [domain] molecular biology [deg 420] --inv. in_field--> [paper] Molecular Dynamics Simulation for All [deg 4]
- OpenIE seeds: 4 (degrees median 3); route: splice variants [deg 3] --inv. contains--> rna seq [deg 81] --inv. is the type of data used in--> rna [deg 135] --connects to--> proteins [deg 87] --is_mentioned_in--> [paper] Molecular Dynamics Simulation for All [deg 14]

### sir4_physics: counts

frame top-5 while OpenIE past 50: 24 queries; OpenIE top-5 while frame past 50: 43 queries (cross-field: 3 vs 2)

### sir4_physics: frame graph finds it, OpenIE buries it

**10.48550_arxiv.2601.20839** (same-field) | walk rank of the gold: SciAffordGraph 3, OpenIE 1292
- query: How can the Friedmann equations be derived from thermodynamics when the apparent horizon is not comoving and matter creation occurs, and what implications does this have for the generalized second law and cosmic acceleration? Prior thermodynamic derivations of the Friedmann equations applied the uni
- gold: Thermodynamics and cosmology.
- SciAffordGraph seeds: 30 (degrees median 4); route: [task] thermodynamic modeling in cosmology [deg 6] --inv. addresses--> [paper] Thermodynamics and cosmology. [deg 2]
- OpenIE seeds: 9 (degrees median 3); route: no route within 4 hops

**10.1103_jrmj-hfwy** (cross-field) | walk rank of the gold: SciAffordGraph 2, OpenIE 1312
- query: How can we efficiently construct total angular momentum eigenstates for many-nucleon systems in spherical shell models, avoiding the complexity of traditional J-scheme and the inefficiency of m-scheme? Existing methods for constructing many-body angular momentum eigenstates either use complicated an
- gold: Quantum Theory of Angular Momentum. This is the most complete handbook on the quantum theory of angular momentum. Containing basic definitions and theorems as well as relations, tables of formula and numerical tables whi
- SciAffordGraph seeds: 36 (degrees median 4); route: [task] quantum theory of angular momentum [deg 4] --inv. addresses--> [paper] Quantum Theory of Angular Momentum [deg 2]
- OpenIE seeds: 9 (degrees median 3); route: no route within 4 hops

**10.1103_w864-hdyw** (same-field) | walk rank of the gold: SciAffordGraph 2, OpenIE 804
- query: How do quantum backreaction effects, particularly cross-correlations between quantum fluctuations of different degrees of freedom, modify the classical bouncing solutions in anisotropic Brans-Dicke cosmology? Prior effective quantum cosmological approaches, applied to isotropic and some anisotropic 
- gold: Singularity removal in a quantum effective evolution of the mixmaster cosmological model. In this work we analyze the evolution of the quantum mixmaster cosmological model within an effective approach. In particular, we 
- SciAffordGraph seeds: 21 (degrees median 7); route: [task] quantum cosmological model evolution [deg 6] --inv. addresses--> [paper] Singularity removal in a quantum effective evolution of the mixmaster  [deg 3]
- OpenIE seeds: 6 (degrees median 4); route: quantum fluctuations [deg 37] --inv. scales with--> entanglement entropy [deg 60] --is_mentioned_in--> [paper] Deformed Fredkin spin chain with extensive entanglement [deg 30] --inv. is_mentioned_in--> hamiltonian [deg 142] --is_mentioned_in--> [paper] Singularity removal in a quantum effective evolution of the mixmaster  [deg 12]

### sir4_physics: OpenIE finds it, frame graph buries it

**10.3390_astronomy5010004** (same-field) | walk rank of the gold: SciAffordGraph 773, OpenIE 1631
- query: Can the flatness, horizon, and primordial perturbation puzzles of the Hot Big Bang be explained as consequences of observing a tiny causal patch of a much larger spacetime, with the positive cosmological constant providing a natural observational cutoff, rather than requiring a dynamical acceleratin
- gold: Information Theory and Statistical Mechanics. Information theory provides a constructive criterion for setting up probability distributions on the basis of partial knowledge, and leads to a type of statistical inference 
- SciAffordGraph seeds: 33 (degrees median 5); route: no route within 4 hops
- OpenIE seeds: 6 (degrees median 30); route: standard model [deg 89] --inv. are consistent with--> measurements [deg 68] --inv. is influenced by--> quantum entanglement [deg 30] --is analyzed using--> statistical mechanics [deg 30] --is_mentioned_in--> [paper] Information Theory and Statistical Mechanics [deg 20]

**10.21468_scipostphys.19.5.125** (same-field) | walk rank of the gold: SciAffordGraph 638, OpenIE 902
- query: How can the Lund string fragmentation function be extracted from hadronization data when the string system contains gluons, given the increased complexity and information loss? Previous methods for extracting the fragmentation function from data either assumed a parametric form or were limited to si
- gold: ON THE PROBLEM OF THE MOST EFFICIENT TESTS OF STATISTICAL HYPOTHESES.
- SciAffordGraph seeds: 36 (degrees median 6); route: [method] traditional parametric methods [deg 8] --inv. is_variant_of--> [method] traditional statistical methods [deg 23] --is_variant_of--> [method] traditional statistical tests [deg 9] --inv. is_variant_of--> [method] efficient statistical tests [deg 14] --inv. contributes--> [paper] ON THE PROBLEM OF THE MOST EFFICIENT TESTS OF STATISTICAL HYPOTHESES. [deg 3]
- OpenIE seeds: 6 (degrees median 4); route: no route within 4 hops

**10.48550_arxiv.2603.04178** (same-field) | walk rank of the gold: SciAffordGraph 600, OpenIE 2
- query: How does the single-fluid magnetohydrodynamic approximation compare with the multi-fluid model in describing the propagation of torsional Alfvén waves from the photosphere to the corona in a partially ionized solar atmosphere? Prior studies of Alfvén wave propagation in the partially ionized solar a
- gold: Torsional Alfvén waves in partially ionized solar plasma: effects of neutral helium and stratification. Context. Ion-neutral collisions may lead to the damping of Alfvén waves in chromospheric and prominence plasmas. Neu
- SciAffordGraph seeds: 26 (degrees median 5); route: [method] three-fluid dynamics model [deg 12] --inv. is_variant_of--> [method] three-fluid ohm's law [deg 12] --inv. contributes--> [paper] Three‐fluid Ohm's law [deg 3] --in_field--> [domain] plasma physics [deg 42] --inv. in_field--> [paper] Torsional Alfvén waves in partially ionized solar plasma: effects of n [deg 3]
- OpenIE seeds: 8 (degrees median 4); route: torsional alfv n waves [deg 4] --is_mentioned_in--> [paper] Torsional Alfvén waves in partially ionized solar plasma: effects of n [deg 14]

### sir4_matsci: counts

frame top-5 while OpenIE past 50: 16 queries; OpenIE top-5 while frame past 50: 16 queries (cross-field: 6 vs 6)

### sir4_matsci: frame graph finds it, OpenIE buries it

**10.1088_1361-6463_ae1b18** (same-field) | walk rank of the gold: SciAffordGraph 2, OpenIE 667
- query: What is the microscopic mechanism that gives rise to the high tunneling magnetoresistance in hcp-Co/h-BN/hcp-Co(0001) magnetic tunnel junctions, and how can it be controlled by electrode composition and interfacial distance? Earlier magnetic tunnel junctions used amorphous oxide barriers, where cond
- gold: Resonant electronic states and<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML" display="inline"><mml:mrow><mml:mi>I</mml:mi><mml:mtext>−</mml:mtext><mml:mi>V</mml:mi></mml:mrow></mml:math>curves of Fe/MgO/Fe(100)
- SciAffordGraph seeds: 41 (degrees median 6); route: [task] tunnel magnetoresistance analysis [deg 9] --inv. addresses--> [paper] Resonant electronic states and<mml:math xmlns:mml="http://www.w3.org/1 [deg 4]
- OpenIE seeds: 12 (degrees median 7); route: graphene [deg 100] --is studied with--> density functional theory [deg 153] --is_mentioned_in--> [paper] Resonant electronic states and<mml:math xmlns:mml="http://www.w3.org/1 [deg 16]

**10.1103_g32j-hnvz** (same-field) | walk rank of the gold: SciAffordGraph 4, OpenIE 450
- query: What are the intrinsic spin and orbital magnetization contributions in the altermagnetic ground state of α-MnTe, and how do they depend on carrier doping? Prior studies on altermagnets have focused on the spin-split electronic structure and the anomalous Hall effect, but the orbital contribution to 
- gold: Orbital magnetization in crystalline solids: Multi-band insulators, Chern insulators, and metals. We derive a multi-band formulation of the orbital magnetization in a normal periodic insulator (i.e., one in which the Che
- SciAffordGraph seeds: 27 (degrees median 5); route: [task] orbital magnetization calculation [deg 11] --inv. addresses--> [paper] Orbital magnetization in crystalline solids: Multi-band insulators, Ch [deg 3]
- OpenIE seeds: 5 (degrees median 17); route: spin orbit coupling [deg 54] --is_mentioned_in--> [paper] Orbital Mechanisms of Electron-Spin Manipulation by an Electric Field [deg 14] --inv. is_mentioned_in--> 2d [deg 34] --is_mentioned_in--> [paper] Orbital magnetization in crystalline solids: Multi-band insulators, Ch [deg 22]

**10.48550_arxiv.2603.04623** (same-field) | walk rank of the gold: SciAffordGraph 5, OpenIE 527
- query: How can we reliably transfer large-area 2D material monolayers and their heterostructures onto both flat and nanostructured substrates while preserving material quality and enabling integration with patterned surfaces? Existing transfer methods for 2D materials include dry transfer using polymer sta
- gold: Optimization of atmospheric plasma treatment of LDPE films: influence on adhesive properties and ageing behavior. One of the major disadvantages of low density polyethylene (LDPE) films is their poor adhesive properties.
- SciAffordGraph seeds: 32 (degrees median 5); route: [limitation] poor adhesive properties [deg 1] --inv. overcomes--> [method] atmospheric plasma treatment [deg 9] --inv. contributes--> [paper] Optimization of atmospheric plasma treatment of LDPE films: influence  [deg 4]
- OpenIE seeds: 14 (degrees median 2); route: no route within 4 hops

### sir4_matsci: OpenIE finds it, frame graph buries it

**10.1038_s41467-026-69659-8** (same-field) | walk rank of the gold: SciAffordGraph 422, OpenIE 123
- query: Can half-integer thermal conductance be realized in a system based on Abelian phases, without underlying non-Abelian topology? Prior methods for achieving half-integer thermal conductance have relied on non-Abelian topological phases, such as fractional quantum Hall states hosting Majorana modes. Th
- gold: Edge mixing dynamics in graphene p–n junctions in the quantum Hall regime. Massless Dirac electron systems such as graphene exhibit a distinct half-integer quantum Hall effect, and in the bipolar transport regime co-prop
- SciAffordGraph seeds: 22 (degrees median 7); route: [function] discern the relationship between electrical and thermal con [deg 7] --inv. achieves--> [method] quantum critical universality in graphene transport [deg 11] --inv. contributes--> [paper] Universality in quantum critical flow of charge and heat in ultra-clea [deg 4] --in_field--> [domain] condensed matter physics [deg 190] --inv. in_field--> [paper] Edge mixing dynamics in graphene p–n junctions in the quantum Hall reg [deg 4]
- OpenIE seeds: 6 (degrees median 5); route: fractional quantum hall states [deg 5] --is_mentioned_in--> [paper] Half-quantized Hall plateaus in the confined geometry of graphene [deg 16] --inv. is_mentioned_in--> graphene [deg 100] --is_mentioned_in--> [paper] Edge mixing dynamics in graphene p–n junctions in the quantum Hall reg [deg 17]

**10.48550_arxiv.2602.18257** (cross-field) | walk rank of the gold: SciAffordGraph 422, OpenIE 335
- query: What are the microscopic mechanisms governing the formation of quadrupolar order and its coupling to magnetic and structural degrees of freedom in 5d1 double perovskites, and what causes the differing behaviors of Ba2MgReO6 and Ba2NaOsO6? Previous first-principles electronic structure calculations h
- gold: Exploring energy landscapes of charge multipoles using constrained density functional theory. We present a method to constrain local charge multipoles within density-functional theory. Such multipoles quantify the anisot
- SciAffordGraph seeds: 30 (degrees median 5); route: [method] first-principles calculations [deg 87] --builds_on--> [method] density functional theory calculations [deg 118] --inv. builds_on--> [method] constrained density functional theory for charge multipoles [deg 11] --inv. contributes--> [paper] Exploring energy landscapes of charge multipoles using constrained den [deg 3]
- OpenIE seeds: 8 (degrees median 4); route: spin orbit coupling [deg 54] --inv. can be enhanced by--> graphene [deg 100] --is studied with--> density functional theory [deg 153] --is_mentioned_in--> [paper] Exploring energy landscapes of charge multipoles using constrained den [deg 17]

**10.48550_arxiv.2602.09844** (same-field) | walk rank of the gold: SciAffordGraph 412, OpenIE 1
- query: How do exciton binding energies and fine-structure splittings depend on the crystal structure and lateral size in CdSe nanoplatelets? Previous approaches for computing exciton properties in quantum-confined structures have used many-body perturbation methods that include electron-hole Coulomb and ex
- gold: Model dielectric function for 2D semiconductors including substrate screening. Dielectric screening of excitons in 2D semiconductors is known to be a highly non-local effect, which in reciprocal space translates to a str
- SciAffordGraph seeds: 27 (degrees median 7); route: [limitation] limited to bulk analysis [deg 6] --inv. same_limitation--> [limitation] less accessible for rapid analysis [deg 2] --inv. limited_by--> [method] numerical ab initio methods [deg 9] --inv. improves_on--> [method] analytical model dielectric function [deg 12] --inv. contributes--> [paper] Model dielectric function for 2D semiconductors including substrate sc [deg 3]
- OpenIE seeds: 15 (degrees median 4); route: dielectric screening [deg 6] --is_mentioned_in--> [paper] Model dielectric function for 2D semiconductors including substrate sc [deg 16]

### tomato: counts

frame top-5 while OpenIE past 50: 169 queries; OpenIE top-5 while frame past 50: 90 queries (cross-field: 17 vs 4)

### tomato: frame graph finds it, OpenIE buries it

**2025_40370301::0** (same-field) | walk rank of the gold: SciAffordGraph 3, OpenIE 2575
- query: What molecular mechanism underlies sAβ-induced impairment of glucose transport across the polarized blood-brain barrier endothelium? Alzheimer’s disease (AD) is characterized by reduced cerebral glucose uptake, detected via $^{18}$FDG-PET imaging, which correlates with cognitive decline and diminish
- gold: Phosphorylation of TXNIP by AKT Mediates Acute Influx of Glucose in Response to Insulin. Growth factors, such as insulin, can induce both acute and long-term glucose uptake into cells. Apart from the rapid, insulin-induc
- SciAffordGraph seeds: 51 (degrees median 4); route: [task] acute glucose influx regulation [deg 5] --inv. addresses--> [paper] Phosphorylation of TXNIP by AKT Mediates Acute Influx of Glucose in Re [deg 4]
- OpenIE seeds: 10 (degrees median 16); route: alzheimer s disease [deg 400] --inv. is associated with--> oxidative stress [deg 109] --inv. is a sensor of--> txnip [deg 13] --is_mentioned_in--> [paper] Phosphorylation of TXNIP by AKT Mediates Acute Influx of Glucose in Re [deg 15]

**2025_41068782::1** (same-field) | walk rank of the gold: SciAffordGraph 1, OpenIE 2554
- query: How do somatic mutations in brain cells contribute to the pathogenesis and progression of sporadic Alzheimer’s disease, given the limitations of germline genetics and environmental factors alone in explaining neurodegeneration? Alzheimer’s disease (AD) is primarily characterized by amyloid-beta plaq
- gold: Signatures of mutational processes in human cancer All cancers are caused by somatic mutations; however, understanding of the biological processes generating these mutations is limited. The catalogue of somatic mutations
- SciAffordGraph seeds: 84 (degrees median 6); route: [task] understanding mutational processes in cancer [deg 5] --inv. addresses--> [paper] Signatures of mutational processes in human cancer All cancers are cau [deg 4]
- OpenIE seeds: 14 (degrees median 4); route: neurons [deg 205] --relay--> information [deg 14] --is_mentioned_in--> [paper] Evidence for an Active Handoff between Hemispheres during Target Track [deg 23] --inv. is_mentioned_in--> results [deg 128] --is_mentioned_in--> [paper] Signatures of mutational processes in human cancer All cancers are cau [deg 19]

**2025_40214960::0** (cross-field) | walk rank of the gold: SciAffordGraph 5, OpenIE 2472
- query: How does risk-reward decision-making vary within and between dementia subtypes (behavioural-variant frontotemporal dementia (bvFTD) and Alzheimer’s disease (AD)), and how is this variability linked to co-occurring clinical symptoms (disinhibition and apathy) and distinct patterns of brain atrophy? P
- gold: Evaluation of a behavioral measure of risk taking: the Balloon Analogue Risk Task (BART). The present study (N = 86) sought to evaluate a laboratory-based behavioral measure of risk taking (the Balloon Analogue Risk Task
- SciAffordGraph seeds: 66 (degrees median 5); route: [task] risk taking assessment [deg 4] --inv. addresses--> [paper] Evaluation of a behavioral measure of risk taking: the Balloon Analogu [deg 3]
- OpenIE seeds: 11 (degrees median 5); route: alzheimer s disease [deg 400] --inv. is affected in--> hippocampus [deg 171] --is a region in--> cud [deg 14] --correlates with--> impulsivity [deg 4] --is_mentioned_in--> [paper] Evaluation of a behavioral measure of risk taking: the Balloon Analogu [deg 14]

### tomato: OpenIE finds it, frame graph buries it

**2025_40999238::1** (same-field) | walk rank of the gold: SciAffordGraph 1515, OpenIE 4
- query: How to non-invasively quantify volumetric pulsatility across cortical layers and white matter in human cerebral microvasculature in vivo? Arterial pulsatility is typically measured using transcranial Doppler ultrasonography and phase-contrast MRI (PC-MRI). These methods primarily quantify velocity p
- gold: Image-based method for retrospective correction of physiological motion effects in fMRI: RETROICOR. Respiration effects and cardiac pulsatility can induce signal modulations in functional MR image time series that increa
- SciAffordGraph seeds: 76 (degrees median 5); route: no route within 4 hops
- OpenIE seeds: 16 (degrees median 4); route: cardiac pulsatility [deg 2] --is_mentioned_in--> [paper] Image-based method for retrospective correction of physiological motio [deg 11]

**2025_41105712::0** (cross-field) | walk rank of the gold: SciAffordGraph 1460, OpenIE 2
- query: How to construct a patient-specific 3D human brain model that:  
- Integrates all six CNS cell types with independently controllable differentiation;  
- Recapitulates functional hallmarks (neurovascular units, myelination, BBB, neuronal activity);  
- Enables systematic dissection of cell-type-spec
- gold: Synthetic alternatives to Matrigel. Matrigel, a basement-membrane matrix extracted from Engelbreth-Holm-Swarm mouse sarcomas, has been used for more than four decades for a myriad of cell culture applications. However, M
- SciAffordGraph seeds: 105 (degrees median 6); route: [limitation] limited functional maturity of hpsc-derived neurons [deg 1] --inv. overcomes--> [method] gentonik [deg 10] --inv. contributes--> [paper] Combined small-molecule treatment accelerates maturation of human plur [deg 5] --in_field--> [domain] cell biology [deg 105] --inv. in_field--> [paper] Synthetic alternatives to Matrigel [deg 3]
- OpenIE seeds: 7 (degrees median 24); route: matrigel [deg 15] --is_mentioned_in--> [paper] Synthetic alternatives to Matrigel [deg 16]

**2025_41031565::1** (same-field) | walk rank of the gold: SciAffordGraph 1446, OpenIE 3
- query: What causes white matter abnormalities in ZDHHC9-associated X-linked intellectual disability? - XLID Pathogenesis: Established models attribute XLID primarily to neuronal dysfunction, such as MECP2 mutations impairing synaptic function (Amir et al., 1999).  
- ZDHHC9 Clinical Phenotypes: Human mutat
- gold: DHHC9 and GCP16 constitute a human protein fatty acyltransferase with specificity for H- and N-Ras. Covalent lipid modifications mediate the membrane attachment and biological activity of Ras proteins. All Ras isoforms a
- SciAffordGraph seeds: 68 (degrees median 6); route: [mechanism] post-translational modifications [deg 1] --inv. works_via--> [finding] conformational changes in protein structures [deg 5] --inv. reports--> [paper] Changes in Protein Structural Motifs upon Post-Translational Modificat [deg 4] --in_field--> [domain] biochemistry [deg 31] --inv. in_field--> [paper] DHHC9 and GCP16 constitute a human protein fatty acyltransferase with  [deg 3]
- OpenIE seeds: 15 (degrees median 2); route: zdhhc9 [deg 2] --is_mentioned_in--> [paper] DHHC9 and GCP16 constitute a human protein fatty acyltransferase with  [deg 22]

### mir: counts

frame top-5 while OpenIE past 50: 15 queries; OpenIE top-5 while frame past 50: 9 queries (cross-field: 0 vs 0)

### mir: frame graph finds it, OpenIE buries it

**ABC_fca75d394e9f7007e1f674c7b99794_27** (same-field) | walk rank of the gold: SciAffordGraph 2, OpenIE 4144
- query: The research focuses on multi-party linguistic entrainment, specifically exploring the relationship between entrainment and team characteristics like gender diversity, team size, and diversity, and its impact on perceived team social outcomes. The abstract doesn't explicitly state the motivations be
- gold: When interacting individuals entrain , they begin to speak more like each other. To support research on entrainment in cooperative multi-party dialogues, we have created a corpus where teams of three or four speakers pla
- SciAffordGraph seeds: 13 (degrees median 2); route: [task] team entrainment in cooperative dialogues [deg 4] --inv. addresses--> [paper] When interacting individuals entrain , they begin to speak more like e [deg 4]
- OpenIE seeds: 6 (degrees median 6); route: effectiveness [deg 44] --is shown through--> experimental results [deg 403] --are related to--> sentiment analysis [deg 254] --is performed using--> corpus [deg 160] --is_mentioned_in--> [paper] When interacting individuals entrain , they begin to speak more like e [deg 11]

**ABC_fe1d6ca4a88c03cfb2ae94ef45030d_44** (same-field) | walk rank of the gold: SciAffordGraph 5, OpenIE 3318
- query: The research problem is the generation of accurate and diverse keyphrases for scientific articles. The research is motivated by the desire to improve keyphrase generation for scientific articles. The authors also aim to contribute to the field by making their implementation publicly available.
- gold: Existing keyphrase generation studies suffer from the problems of generating duplicate phrases and deﬁcient evaluation based on a ﬁxed number of predicted phrases. We propose a recurrent generative model that generates m
- SciAffordGraph seeds: 13 (degrees median 8); route: [task] keyphrase generation [deg 13] --inv. addresses--> [paper] Existing keyphrase generation studies suffer from the problems of gene [deg 3]
- OpenIE seeds: 4 (degrees median 14); route: implementation [deg 20] --inv. introduce--> we [deg 589] --compared--> our model [deg 229] --is_mentioned_in--> [paper] Existing keyphrase generation studies suffer from the problems of gene [deg 17]

**ABC_ffd65a1a02c852a2670b471fb4b110_10** (same-field) | walk rank of the gold: SciAffordGraph 1, OpenIE 2609
- query: The main research problem is the challenge of modeling physical plausibility, specifically the failure of distributional methods in supervised settings. This work aims to investigate whether large pretrained language models can overcome this limitation and learn to model physical plausibility direct
- gold: Distributional data tells us that a man can swallow candy, but not that a man can swallow a paintball, since this is never attested. However both are physically plausible events. This paper introduces the task of semanti
- SciAffordGraph seeds: 15 (degrees median 5); route: [task] semantic plausibility recognition [deg 2] --inv. addresses--> [paper] Distributional data tells us that a man can swallow candy, but not tha [deg 4]
- OpenIE seeds: 3 (degrees median 12); route: natural language understanding [deg 78] --inv. is used for--> deep learning [deg 282] --is reviewed in--> this paper [deg 1188] --is_mentioned_in--> [paper] Distributional data tells us that a man can swallow candy, but not tha [deg 18]

### mir: OpenIE finds it, frame graph buries it

**ABC_8dbc779d455ad72def6654564f9e13_43** (same-field) | walk rank of the gold: SciAffordGraph 2619, OpenIE 4
- query: The research problem is the inefficiency and high cost of manual transcription for language documentation initiatives.  The study aims to investigate the impact of language on automatic approaches for language documentation, specifically exploring the effectiveness of bilingual-rooted unsupervised w
- gold: We present a first attempt to perform attentional word segmen-tation directly from the speech signal, with the final goal to automatically identify lexical units in a low-resource, unwritten language (UL). Our methodolog
- SciAffordGraph seeds: 21 (degrees median 6); route: no route within 4 hops
- OpenIE seeds: 4 (degrees median 4); route: language documentation [deg 5] --is_mentioned_in--> [paper] We present a first attempt to perform attentional word segmen-tation d [deg 15]

**ABC_d1dce63d89e8cfc73962413734bf7b_2** (same-field) | walk rank of the gold: SciAffordGraph 659, OpenIE 1
- query: The research addresses the challenge of applying semantic specialization, a technique for improving distributional vectors, to languages lacking complete external linguistic resources like WordNet. The primary motivation is to enable semantic specialization in languages lacking complete external res
- gold: Semantic specialization is a process of fine-tuning pre-trained distributional word vectors using external lexical knowledge (e.g., WordNet) to accentuate a particular semantic relation in the specialized vector space. W
- SciAffordGraph seeds: 21 (degrees median 6); route: [task] distributional semantic modeling [deg 14] --inv. addresses--> [paper] We present SimLex-999, a gold standard resource for evaluating distrib [deg 3] --in_field--> [domain] natural language processing [deg 2358] --inv. in_field--> [paper] Semantic specialization is a process of fine-tuning pre-trained distri [deg 4]
- OpenIE seeds: 5 (degrees median 3); route: wordnet [deg 82] --is_mentioned_in--> [paper] Semantic specialization is a process of fine-tuning pre-trained distri [deg 25]

**ABC_24ee9b2bd8c97cbe923bc747b09806_12** (same-field) | walk rank of the gold: SciAffordGraph 595, OpenIE 3143
- query: The main research problem is the limitation of existing computational language learning models that require text, while humans learn from direct speech interaction. The research aims to develop a visually grounded sentence encoder that can learn language directly from speech. The research is motivat
- gold: Recent advances in Neural Machine Translation (NMT) show that adding syntactic information to NMT systems can improve the quality of their translations. Most existing work utilizes some specific types of linguistically-i
- SciAffordGraph seeds: 21 (degrees median 6); route: [task] spoken language understanding [deg 27] --inv. addresses--> [paper] In practice, most spoken language understanding systems process user i [deg 3] --in_field--> [domain] natural language processing [deg 2358] --inv. in_field--> [paper] Recent advances in Neural Machine Translation (NMT) show that adding s [deg 3]
- OpenIE seeds: 5 (degrees median 2); route: speech [deg 26] --inv. is suited to work from--> approach [deg 428] --uses--> recurrent neural networks [deg 229] --inv. is a type of--> rnn [deg 228] --is_mentioned_in--> [paper] Recent advances in Neural Machine Translation (NMT) show that adding s [deg 11]

