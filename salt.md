# Selective Annotation: Certified Accuracy, Minimal Human Cost

**Paper ID: 667**

---

## Abstract

This paper studies large-scale dataset labeling via selective annotation. Given an unlabeled dataset $D$, the goal is to label all samples in $D$ with minimal human annotation cost, while ensuring a certified labeling accuracy. We show that this problem is intractable. We propose SALT (Selective Annotation for Labeling and Training), a human-in-the-loop framework for labeling. SALT incrementally trains a model $M$ and introduces a novel curvature-adaptive coverage condition under which $M$ is provably guaranteed to meet a target accuracy bound on $D$. It iteratively selects a small number of high-impact samples for annotation, guided by a unified coverage–uncertainty criterion derived from the theoretical bound. SALT further integrates multiple labeling functions through a logic-based correction mechanism to mitigate weak supervision noise, and leverages these enhanced weak signals to speed up early learning when annotated data is scarce. Using real-world datasets, we empirically show that SALT is up to 72.4% more accurate than active learning methods, saving human effort by over 75.7%.

---

## 1 Introduction

Modern machine learning (ML) systems, particularly large language models (LLMs), rely on large-scale labeled datasets. Data labeling, i.e., assigning predefined categories or annotations to raw samples, provides the supervision signal necessary for training, instruction tuning, and alignment. High-quality labeled datasets play a critical role in a wide range of applications such as autonomous driving, social media intent recognition and medical diagnosis [1], and their importance continues to grow as ML systems scale.

In practice, training a production-grade ML model often requires a labeled dataset $D$ containing millions of samples. For example, ImageNet [20] includes over 14 million human-annotated images collected over several years. Similarly, natural language processing systems depend on large-scale labeled corpora, and modern commercial LLMs are trained or tuned on tens to hundreds of millions of annotated text instances for tasks such as sentiment analysis, question answering, and instruction execution [28, 53, 78].

However, constructing such datasets relies heavily on human annotators, making it costly and time-consuming. Labeling one million text documents requires over 1,388 person-hours even under optimistic assumptions, and expert annotation can cost $50–100 per instance [31]. In many domains, qualified annotators are scarce; diverting clinicians from patient care to label data is neither feasible nor sustainable. This highlights the need for labeling strategies that minimize human effort while preserving accuracy and reliability.

### Selective Annotation

The question is: Can we label an entire dataset by annotating only a small subset, while provably meeting a target accuracy? This gives rise to the selective annotation problem.

Given an unlabeled dataset $D$, the goal is to manually annotate only a small subset $D_S \subseteq D$ and automatically label the remaining samples, while meeting a prescribed target accuracy threshold over the entire dataset. The objective is to minimize human annotation effort while guaranteeing labeling quality. For simplicity, we focus on classification, one of the most common labeling tasks in practice.

A natural approach is to train a labeling model $M$ that annotates samples in $D$ with provable accuracy. However, no prior work targets this objective and thus falls short:

1. **Active learning** [10, 39, 74] reduces annotation cost but optimizes generalization rather than dataset-wide labeling accuracy on $D$, and assumes a fully labeled validation or test set for selection and stopping.
2. **Weak supervision** [8, 55, 58] scales labeling via heuristic signals, which are often noisy and can degrade model quality.
3. **Semi-supervised learning** [12, 67] expands labels via pseudo-labels, but errors can propagate and amplify over iterations.

This gap leads to three challenges:

- **Certified accuracy is missing.** No existing approach provides verifiable guarantees that the labeled dataset meets a target accuracy threshold, as true labels are unknown during the process.
- **Minimizing human effort is hard.** Selecting a subset $D_S \subseteq D$ that maximizes global impact is inherently combinatorial and intractable (Section 3), since the value of each sample depends on its interaction with others and the underlying data geometry.
- **Noise must be explicitly controlled.** Weak supervision improves scalability but introduces noise that, if uncorrected, can propagate and violate the objective accuracy bound.

### SALT

The key idea is to view selective annotation as a coverage problem in embedding space: each annotated sample covers a local region where the model's loss is provably bounded, and achieving global coverage over $D$ yields a dataset-wide accuracy guarantee.

Based on this idea, we propose SALT (Selective Annotation for Labeling and Training), a framework that addresses the three challenges above: certified accuracy, minimal human effort, and noise tolerance. SALT trains a classification model $M$ to automatically label $D$ while minimizing human cost under a given accuracy target. SALT employs an iterative human-in-the-loop (HITL) procedure. At each round, it selects a subset $D_t \subseteq D$ for annotation, updates $M$, and checks a coverage-based certification condition. The process terminates once the induced coverage spans $D$, at which point $M$ is guaranteed to meet the given accuracy target on the entire dataset.

![Figure 1: Prior methods vs. SALT](https://aka.doubaocdn.com/s/imxf1wfD51)

*Figure 1: Prior methods vs. SALT. (a) Active learning (e.g., Entropy [74]) optimizes generalization via uncertainty sampling, but offers no dataset-level guarantee on $D$ and relies on heuristic stopping (e.g., budget or validation proxy). (b) Weak supervision (e.g., FlyingSquid [25]) optimizes label aggregation from heuristic sources, but provides neither control over global accuracy due to noise nor well-defined stopping criterion. (c) SALT induces a coverage objective aligned with dataset labeling, provides a certified accuracy guarantee via curvature-adaptive coverage, and admits an explicit stopping rule: it terminates once the certified regions fully cover $D$.*

At a high level, SALT makes the following contributions:

1. **Certified selective annotation.** We introduce the first framework for selective annotation with certified dataset-wide accuracy. We derive curvature-adaptive coverage, where the size of the local cover induced by each annotated sample is determined by the local loss geometry. When these local covers jointly span the dataset, i.e., a full coverage, we obtain a provable bound on global risk and labeling accuracy. This provides a verifiable stopping criterion, unlike prior methods that rely on heuristics without guarantees.

2. **Coverage-driven selection.** We develop a selection strategy that directly optimizes certified coverage, rather than heuristic uncertainty or diversity. At each round, SALT selects hard samples that expand uncovered certified regions, progressing from global coverage to boundary refinement. This unifies uncertainty and diversity under a single objective, aligned with the accuracy guarantee.

3. **Integration of weak signals.** We incorporate weak supervision via a logic-based correction framework that controls noise without compromising guarantees. By resolving conflicts among labeling functions using rules learned from human annotations, SALT produces higher-quality labels for efficient bootstrapping. This reduces human effort while preserving the certified accuracy guarantee.

4. **Experimental evaluation.** Using real-world datasets, we empirically validate the effectiveness and efficiency of SALT. We find the following:
   - SALT achieves the highest labeling accuracy, improving over active learning baselines by 25.9–72.4%.
   - It reduces human annotation cost by up to 75.7% under high-accuracy targets.
   - Its weak-supervision module improves label quality by up to 7.8% and further reduces annotation cost via effective bootstrapping.
   - It speeds up convergence and reduces end-to-end execution time by >44.0%, verifying strong efficiency and robustness.

**Organization.** Section 2 reviews related work on data labeling. Section 3 formulates the selective annotation problem, proves its intractability, and presents the SALT framework. Section 4 introduces the curvature-adaptive cover and derives the corresponding accuracy guarantee. Section 5 presents the coverage-driven selection strategy, and Section 6 describes the logic-based integration of weak supervision. Section 7 reports experimental results, and Section 8 concludes with a discussion of novelty and future directions.

---

## 2 Related Work

This section categorizes related work on data labeling.

### Active Learning

Active learning aims to train a model $M$ with strong generalization while minimizing annotation cost. In the standard pool-based setting, the learner iteratively selects batches of unlabeled samples for annotation, augments the training set, and updates $M$ to guide subsequent selection.

Prior selection strategies fall into three categories:

1. **Uncertainty-based methods** (e.g., Uncertainty [46], Margin [62, 68], Entropy [74], Least Confidence [70] and others [26, 42, 71]) prioritize samples on which the current model is most uncertain.

2. **Diversity-based methods** (e.g., CoreSet [64], TypiClust [41], and others [9, 17, 41, 65, 76, 79]) select representative subsets that cover the input space and reduce redundancy.

3. **Hybrid methods** (e.g., BADGE [4], UHerding [10], ALFA-Mix [54], and others [39, 47]) combine uncertainty and diversity to balance informativeness and coverage, particularly in batch-mode settings.

Our setting differs from active learning in the following:

1. Rather than optimizing generalization to unseen data, SALT targets labeling accuracy on the given dataset $D$.
2. SALT provides a coverage-based theoretical guarantee on labeling accuracy, whereas active learning typically lacks such guarantees.
3. SALT employs a selection strategy explicitly driven by coverage, adaptively balancing multiple criteria instead of relying on fixed heuristics.
4. SALT integrates multiple labeling functions as additional supervision, which lies outside the standard active learning paradigm.

### Weak Supervision

Weak supervision reduces reliance on large-scale manual annotation by employing labeling functions, e.g., heuristic rules, distant supervision or programmatic functions, to automatically assign labels to unlabeled data. However, such labeling functions are not always available (e.g., in visual tasks), and when they are, they are often noisy, incomplete and conflicting, which limits their reliability.

Existing methods can be grouped into three categories:

1. **Rule- and distant supervision approaches** [8, 21, 27, 33, 55, 58, 59, 69] encode domain knowledge or align data with external sources to heuristically generate labels.

2. **Label aggregation methods** [19, 25, 36] model the accuracy and correlations of multiple noisy sources to deduce latent true labels, typically via generative or probabilistic models.

3. **Joint optimization and noise-aware learning** [56, 60] integrate weak label inference with model training, refining labels through iterative or end-to-end optimization.

SALT optionally integrates weak supervision to speed up learning when labeled data is scarce and reliable labeling functions are available. To ensure robustness, it employs a logic-based correction mechanism to detect and mitigate errors, preventing noise propagation. Unlike purely model-based aggregation, it explicitly reasons over heterogeneous and complementary labeling functions, improving reliability while preserving the overall accuracy guarantee.

### Semi-supervised Learning

Semi-supervised learning [12, 14, 44, 67, 75, 80] trains models using a small labeled set together with a large unlabeled set, typically via pseudo-labeling. These methods exploit data distribution and model confidence to improve generalization, but early errors may be reinforced, and they do not guarantee full-dataset accuracy.

In contrast, selective annotation targets reliable labeling of a fixed dataset under explicit accuracy constraints. Its goal is not to improve generalization via pseudo-label expansion, but to control dataset-wide labeling errors. Accordingly, SALT adopts conservative sample selection and correction mechanisms to limit error accumulation. This difference in objective and risk tolerance distinguishes selective annotation from semi-supervised learning.

### Coreset Selection

Prior work [2, 35, 38–40, 48, 64, 66, 72] selects a small yet representative subset from a large, fully labeled dataset, so that training on the subset reduces computational cost while preserving generalization performance. This differs from selective annotation. Coreset methods assume full label access during selection, enabling label-aware criteria such as class-balanced coverage, gradient matching, and validation-objective optimization. In contrast, SALT selects samples before labels are observed, and must rely solely on pre-label information, and cannot directly use criteria that require ground-truth labels over the dataset.

---

## 3 SALT for Selective Annotation

This section formulates and analyzes the selective annotation problem (Section 3.1), and presents SALT at a high level (Section 3.2).

### 3.1 The Selective Annotation Problem

#### Preliminaries

We start with basic notations and definitions.

**Datasets.** Let $D = \{x_i\}_{i=1}^n$ denote an unlabeled dataset with $n$ samples. Each sample $x_i \in \mathcal{X}$ is a data point in the sample space $\mathcal{X}$, e.g., an image, a document, or a relational tuple, etc.

**Labeling.** We consider classification tasks, where each sample is assigned one or more labels. We focus on w.l.o.g. single-label classification, where each $x_i \in D$ is assigned a label $y_i \in \{1, ..., C\}$ from $C$ classes. Let $D = \{(x_i, y_i)\}_{i=1}^n$ denote the fully labeled dataset derived from $D$. Following [18, 32, 49, 51], we assume an underlying conditional label distribution $\eta(x) = Pr[y | x] \in \Delta^C$, where $\Delta^C$ is the $C$-dimensional probability simplex and $y \sim \eta(x)$.

**Labeling models.** We consider a multi-layer perceptron classifier $M_\theta: \mathcal{X} \to \{1, ..., C\}$, where $\theta$ is the model parameters (the weights and biases). The model consists of multiple fully connected layers followed by a softmax output layer. Given $x \in \mathcal{X}$, let $x \in \mathbb{R}^k$ be its embedding. The model produces logits $z_\theta(x) \in \mathbb{R}^C$, and the class-probability output is $p_\theta(x) = \text{softmax}(z_\theta(x)) \in \Delta^C$. When clear from context, we write $p_\theta$ as $p$ and use $x$ and $x$ interchangeably.

**Labeling accuracy.** Given a labeling model $M_\theta$ and an unlabeled dataset $D$, we define the labeling accuracy of $M_\theta$ using the Normalized Cross-Entropy Score [32], a standard metric for classification:

$$acc(M_\theta, D) := 1 - R_D(\theta) / \ln C \tag{1}$$

Here $R_D(\theta) = \frac{1}{|D|} \sum_{(x, y) \in D} \ell(\theta; x, \eta)$ is the empirical risk over $D$, where $\ell$ is the cross-entropy loss $\ell(\theta; x, \eta) = \sum_{k=1}^C -\eta_k \ln p_k$. Intuitively, this score serves as an accuracy metric with confidence awareness: 0 for random guessing and 1 for perfect predictions.

#### Problem Statement

We now state the selective annotation problem (SAP).

- **Input:** An unlabeled dataset $D = \{x_i\}_{i=1}^n$, a classification model $M$ to train, a learning algorithm $A$, and an accuracy tolerance $\zeta$.
- **Output:** An annotated dataset $D_S$ obtained from a subset $S$ of $D$.
- **Objective:** Minimize the cardinality $|D_S|$ of $D_S$ subject to $acc(M_{\theta_S}, D) \geq acc(M_{\theta_D}, D) - \zeta$, where $\theta_S$ (resp. $\theta_D$) denotes the parameters of $M$ learned by $A$ on $D_S$ (resp. $D$).

Intuitively, SAP aims to select a minimum subset $S \subseteq D$ for human annotation and construct a labeled subset $D_S$. It then trains a labeling model $M_{\theta_S}$ on $D_S$, while requiring its accuracy on $D$ to be within $\zeta$ of that of $M_{\theta_D}$ trained on the fully labeled dataset $D$. This formulation promotes scalable automated labeling by reducing human annotation effort without sacrificing labeling quality.

Despite its appeal, SAP is intractable. Its decision version, denoted by DSAP, is to decide, given a budget $N$ and a tolerance $\zeta$ on the labeling accuracy deviation, whether there exists a subset $S \subseteq D$ such that $|S| \leq N$ and $acc(M_{\theta_S}, D) \geq acc(M_{\theta_D}, D) - \zeta$.

**Theorem 1:** SAP is NP-hard. $\square$

*Proof sketch:* We prove the NP-hardness by reduction from the set cover (see [5] for details), which is NP-complete (cf. [29]). $\square$

In practice, the learned decision boundary of a model $M$ is highly non-convex and depends sensitively on which samples are annotated. SAP is hard because the benefit of annotating a point does not depend on the point alone: it depends on how its label interacts with nearby samples, especially in class-overlap regions. Moreover, both the true label and the resulting gain are revealed only after paying the annotation cost. Hence, SAP is a combinatorial, delayed-feedback search problem, which explains its difficulty.

### 3.2 SALT: An Overview

Despite this intractability, we develop SALT, a human-in-the-loop framework for selective annotation with accuracy guarantees.

![Figure 2: The data labeling workflow with SALT](https://aka.doubaocdn.com/s/cL9T1wfD51)

*Figure 2: The data labeling workflow with SALT.*

#### Integration with Data Labeling Workflows

As shown in Figure 2, given an unlabeled dataset $D$, SALT produces a labeling model $M_{\theta_S}$ trained using an annotated subset $D_S$ from $D$. The model is then used to label the remaining samples in $D$, yielding a complete labeled dataset $D_{\theta_S}$ with certified accuracy. This output can be passed directly to quality inspection, where auditors evaluate a random sample from $D_{\theta_S}$ and issue the final quality report.

This deployment setting also explains why active learning is not a natural fit. Most active learning pipelines rely on a fully labeled validation or test set for selection and stopping, optimizing generalization on that proxy. In production labeling, however, such a reference set for the same dataset $D$ is typically unavailable before labeling, and quality is assessed only after annotation via sampled human review. SALT is designed for this setting: it optimizes quality directly on $D$ and produces labels that can be audited immediately.

#### HITL Pipeline

Figure 2 also shows the human-in-the-loop (HITL) pipeline of SALT. The system iteratively combines human annotations and weak supervision to improve the labeling model. Each round $t$ uniformly performs selective annotation, weak-label refinement, model training and certified coverage checking, in four steps:

1. **Selective human annotation.** Given the current model $M_t$ and its certified coverage over $D$, SALT selects a small batch $D_t \subseteq D$ for human annotation, to expand coverage with minimal labeling cost. Human annotators provide trusted labels to update the model.

   We assume annotators are reliable, as is typical in production settings with qualified and properly incentivized experts. If this assumption does not hold, additional quality-control mechanisms, e.g., cross-validation, can be applied, but are beyond our scope.

2. **Weak label refinement.** If labeling functions are available for the specific labeling task, SALT can incorporate weak supervision to accelerate model training (the blue arrows). It applies them to generate candidate labels, then reconciles disagreements and filters unreliable outputs using the human annotations collected so far, by adopting a logic-based framework with Label Refining Rules. This improves scalability while controlling noise propagation.

3. **Model training.** Using human labels and refined weak labels, SALT updates $M_t$ to $M_{t+1}$. Early rounds rely more on weak labels to get started, while later rounds rely more on human labels to improve quality and support the final guarantee.

4. **Coverage checking.** Finally, SALT checks whether the updated model $M_{t+1}$ satisfies a certified coverage condition. Each annotated sample defines a certified local cover in embedding space. When these local covers together cover $D$, SALT terminates with a dataset-wide accuracy guarantee; otherwise, it proceeds to the next round.

#### Key Properties

SALT addresses the three challenges in SAP:

1. **Certified accuracy.** Through iterative coverage checking and a verifiable termination condition, it provides high-confidence evidence that the final labels on $D$ meet the target labeling accuracy.

2. **Limited human supervision.** Rather than uniformly annotating data, it uses coverage-driven selective annotation to route only the most influential uncovered samples to humans in each round.

3. **Suppressed noise.** By combining weak-label generation with conflict resolution anchored by human-labeled samples, it leverages scalable weak signals while mitigating noise propagation.

---

## 4 Accuracy Guarantees

This section establishes our theoretical foundation for SALT.

### 4.1 Objective and Overview

Given a model $M$ trained on a labeled subset $D_S$, we want to develop a verifiable condition under which $M$ achieves certified dataset-wide labeling accuracy on $D$. This amounts to identifying regions where the model is reliable, by combining local stability with global coverage, which together capture how well $M$ fits the entire dataset. The key idea is to control how far errors can propagate: if every point lies close to a labeled point with stable predictions, then the entire dataset is well controlled.

This requires combining two types of information: (1) local sensitivity of the model around each point, and (2) global coverage of the dataset. Existing active learning criteria capture only one at a time. Uncertainty-based methods [46, 62, 68, 70] focus on local difficulty but do not provide dataset-level guarantees, while diversity-based methods (e.g., $k$-center greedy [64]) capture global spread but ignore that representativeness varies across regions and models. As a result, neither alone yields a verifiable stopping rule for selective annotation, and naive hybrids still treat them as separate heuristics.

The example below illustrates the main challenge: local difficulty and global coverage must be unified in a single object.

**Example 1:** Figure 3 shows a 2D projection of different sampling strategies on a three-class AGNews subset [34]. Given a budget of 64 annotations, (a) uncertainty-based Margin [68] prioritizes points near the decision boundary of the current model; however, it can distort the global geometry, e.g., between the green and yellow classes, especially when the model is poorly initialized. (b) Diversity-based $k$-center greedy [64] captures global geometry better, but does not directly support the decision boundary refinement needed for high accuracy. Consequently, it may over-select points far from informative boundary regions, wasting human annotation effort. $\square$

![Figure 3: Sample selection of different active learning strategies](https://aka.doubaocdn.com/s/us191wfD58)

*Figure 3: Sample selection of different active learning strategies. (a) Margin [62, 68]: 64 annotations. (b) $k$-center greedy [64]: 64 annotations.*

#### Certified Coverage Framework: An Overview

We recast selective annotation as certified coverage in embedding space. Each labeled point defines a local cover, i.e., a neighborhood where the model's loss changes only slightly. Global progress is then measured by the union of these covers. This provides a single geometric view that captures both local reliability and global coverage.

- **Local: curvature-adaptive influence.** A labeled point $x$ certifies the largest neighborhood where the loss stays within a given budget, defining its local region of confidence. Its size is curvature adaptive: in flat regions the loss changes slowly, so the region can expand, but in steep regions small changes can sharply increase loss, so the region must shrink. Thus, points near complex boundaries certify small covers, while points in stable regions certify larger ones.

- **Global: dataset-wide coverage.** We extend this local notion to a global certificate by aggregating all such covers. An unlabeled point that is in at least one cover is covered, i.e., its loss is close to that of a nearby labeled point. Uncovered points indicate where the model lacks guarantees and where new annotations are most useful.

Once all points are fully covered, the desired accuracy guarantee follows from the global risk bound below.

### 4.2 Deriving Certified Coverage

We now formalize this intuition. We view the dataset as embedded points in $\mathbb{R}^k$ and derive the certificate in four steps: (1) bound local loss variation, (2) compute the largest cover radius, (3) aggregate local covers into a dataset-wide bound on the empirical risk gap, and (4) translate the bound into a certified accuracy guarantee and stopping rule. Below we present the main results (see [5] for proofs).

#### (1) Bounded Loss Variations

Let $x \in \mathbb{R}^k$ be an embedded input sample. Write $\eta(x) = Pr[y | x] \in \Delta^C$ for the conditional label distribution, and let $\ell(x, \eta)$ be the cross-entropy loss. Given $x$, the model $M_\theta$ produces logits $z_\theta(x) \in \mathbb{R}^C$ and class probabilities $p_\theta(x) \in \Delta^C$.

For two nearby embedded points $x$ and $x'$, we want to quantify how much the loss can change inside a small neighborhood around $x$. Under a standard smoothness assumption that $\eta(\cdot)$ is Lipschitz continuous [63], the next lemma controls this local loss variation.

**Lemma 2:** Assume that $\eta(x)$ is $\lambda$-Lipschitz continuous under $\|\cdot\|_2$. Fix $x, x' \in \mathbb{R}^k$ with a maximum distance $\|x' - x\|_2 \leq \delta$. Then

$$|\ell(x', \eta') - \ell(x, \eta)| \leq b\delta + a\delta^2 \tag{2}$$

with $a = \frac{1}{2} h \Gamma_\theta^2 + \lambda \Gamma_\theta$, $b = \|\eta - p\|_2 \Gamma_\theta + \lambda\|z\|_2$.

Here $p = p_\theta(x)$, $\Gamma_\theta$ is the product of spectral norms across network $M_\theta$, $h = \|H\|_2$ with $H = \text{diag}(q(\xi)) - q(\xi)q(\xi)^\top$ for some $\xi$ on the segment between $x$ and $x'$. $\square$

Lemma 2 shows that within a ball $B(x, \delta)$ around a labeled point, the loss variation is bounded by $e(x, \delta, \theta) := a\delta^2 + b\delta$. This bound depends on the prediction mismatch $\|\eta - p\|_2$ and the local curvature $h$. They jointly control how far the point can certify its neighborhood.

Intuitively, this bound quantifies how "stable" the model is at $x$: it is small when the model prediction already matches the local label distribution and the loss geometry is smooth, and large when the model is uncertain or the region is highly curved.

*Proof sketch:* Layer-wise Lipschitzness gives $\|z_\theta(x') - z_\theta(x)\|_2 \leq \Gamma_\theta \delta$. A Taylor bound on the cross-entropy term and the $\lambda$-Lipschitz continuity of $\eta(\cdot)$ then yield $|\ell(x', \eta') - \ell(x, \eta)| \leq b\delta + a\delta^2$. $\square$

#### (2) Local Cover Radius

We invert the loss bound to ask: how far can we move from a labeled point while keeping the loss within a tolerable budget? Fix a local loss budget $\epsilon > 0$. The radius at $x$ is the largest radius for which every point inside the ball remains within this $\epsilon$. Equivalently, it is the maximum admissible neighborhood satisfying $e(x, \delta, \theta) \leq \epsilon$. Solving for this yields

$$\delta_x(\epsilon, \theta) = \frac{2\epsilon}{b + \sqrt{b^2 + 4a\epsilon}} \tag{3}$$

This is the key curvature-adaptive quantity in SALT. It depends on the prediction mismatch $\|\eta - p\|_2$ and the curvature term $h$, and thus varies across the embedding space. It unifies confidence and representativeness: when $\|\eta - p\|_2$ is large or $h$ is high, the denominator in Equation 3 gets larger, and $\delta_x$ gets smaller, so the point certifies only a small neighborhood. When $\|\eta - p\|_2$ is small and curvature is mild, the denominator decreases, and $\delta_x$ grows, allowing the point to represent a larger neighborhood.

**Example 2:** Continuing Example 1, Figure 4 shows certified coverage with 24, 44 and 64 annotations. The black circles are now instances of local covers $B(x, \delta_x(\epsilon, \theta))$, with radii determined by the local loss geometry: points in stable regions certify larger neighborhoods, while points near harder regions certify smaller ones. As more points are annotated, a larger fraction of the dataset is covered, making progress toward the stopping rule explicit. $\square$

![Figure 4: Sample selection of SALT in 3 rounds](https://aka.doubaocdn.com/s/RVhb1wfD58)

*Figure 4: Sample selection of SALT in 3 rounds. Newly annotated points are marked in red, with circles indicating their certified local covers. (a) Round 0: 4 initial annotations. (b) Round 1: 20 new samples (24 in total). (c) Round 2: 20 new samples (44 in total). (d) Round 3: 20 new samples (64 in total).*

#### (3) Dataset-wide Aggregation

We now extend pointwise stability to a dataset-wide guarantee. Since each labeled point represents a different number of samples, we aggregate their contributions by weighting each center based on how many points it covers.

Suppose that the annotated set $S \subseteq D$ induces local balls $B(x_i, \delta_{x_i}(\epsilon, \theta))$ that cover $D$, i.e., $D \subseteq \bigcup_{x_i \in S} B(x_i, \delta_{x_i}(\epsilon, \theta))$. To aggregate this cover, let $\tau: D \to S$ be a surjective map that assigns each $x'_j \in D$ to a center $x_i \in S$ such that $x'_j \in B(x_i, \delta_{x_i}(\epsilon, \theta))$. The corresponding weight is

$$w_i = \left|\left\{x'_j \in D \mid \tau(x'_j) = x_i\right\}\right|$$

which counts the points covered by center $x_i$.

The empirical risk over $D$ is $R_D(\theta) = \frac{1}{|D|} \sum_{(x, y) \in D} \ell(x, y)$. To relate this to training on $D_S$, we define the weighted empirical risk over $D_S$:

$$\hat{R}_{D_S}(\theta) = \frac{1}{|D|} \sum_{(x_i, y_i) \in D_S} w_i \ell(x_i, y_i)$$

This assigns each labeled point a weight equal to the number of data points it represents under the cover.

Denote by $\theta_D = \arg\min R_D(\theta)$ and $\theta_S = \arg\min \hat{R}_{D_S}(\theta)$ the empirical risk minimizers (ERM) over labeled datasets $D$ and $D_S$ respectively. Assume uniform bounds $\Gamma_\theta \leq \Gamma$ on the product of layer spectral norms and $\|z(x; \theta)\|_2 \leq B$ on the logit norm.

We next show that complete certified coverage ensures that training on the labeled subset is nearly as effective as training on the full dataset, i.e., the full-data risk of $\theta_S$ remains close to that of $\theta_D$.

**Theorem 3:** Consider a model $M_\theta$ trained over a labeled subset $D_S$. Under the coverage condition $D \subseteq \bigcup_{x_i \in S} B(x_i, \delta_{x_i}(\epsilon, \theta))$, with surjective assignment induced by $\tau$ and weights $w_i$, we have:

$$
\begin{aligned}
|R_D(\theta_S) - R_D(\theta_D)| & \leq \frac{1}{|D|} \sum_{x_i \in S} \left[w_i e(x_i, \delta_{x_i}(\epsilon, \theta), \theta_S)\right. \\
& \left. + w_i \hat{e}(x_i, \delta_{x_i}(\epsilon, \theta), \theta_D)\right],
\end{aligned}
$$

where $\hat{e}(x, \delta, \theta) = (\sqrt{2}\Gamma + \lambda B)\delta + (\frac{1}{4}\Gamma^2 + \lambda\Gamma)\delta^2$. $\square$

*Proof sketch:* We split the risk gap into two deviations between full-data and weighted risks. Applying Lemma 2 under the assignment $\tau$, and using $\|\eta - p\|_2 \leq \sqrt{2}$ and $h \leq 1/2$, we have $\hat{e}$. $\square$

Intuitively, Theorem 3 connects local covers to a global guarantee. If every point in $D$ is covered by a certified ball, then training on the labeled subset, with each center weighted by its coverage, closely matches training on the full dataset. Hence, complete certified coverage ensures that unlabeled data does not introduce additional risk, and as a result, further annotation becomes unnecessary.

The bound gets tighter as the local radii shrink. By Equation 3, $\delta_x(\epsilon, \theta) \to 0$ as $\epsilon \to 0$, and $e$ and $\hat{e}$ scale as $O(\delta + \delta^2)$. Thus, smaller local variation yields a smaller global risk gap.

We can further simplify the upper bound as follows:

**Corollary 4:** Under the coverage condition of Theorem 3, we have:

$$|R_D(\theta_S) - R_D(\theta_D)| \leq 2\epsilon \tag{4}$$

when $\theta_S$ and $\theta_D$ recover the true conditional probabilities on $S$, i.e., $p_{\theta_S}(x_i) = p_{\theta_D}(x_i) = \eta(x_i)$ for every $x_i \in S$. $\square$

Intuitively, this assumes that $M_\theta$ is expressive enough and well-trained to capture the true distribution over $D_S$, which is generally reasonable in practice as long as $M_\theta$ has sufficient capacity.

#### (4) Certified Accuracy

We convert the risk bound into the selective-annotation accuracy target. By $acc(M_\theta, D) = 1 - R_D(\theta) / \ln C$:

$$acc(M_{\theta_D}, D) - acc(M_{\theta_S}, D) \leq \frac{|R_D(\theta_S) - R_D(\theta_D)|}{\ln C}$$

Thus, bounding the risk gap directly bounds the accuracy gap. To enforce slack $\zeta$, it suffices to require the risk gap $\leq \zeta \ln C$. By Corollary 4, this can be done by setting $\epsilon = \frac{1}{2} \zeta \ln C$. Thus, $\epsilon$ is the local loss budget that ensures the desired global accuracy slack.

**Termination condition for SALT.** The stopping rule is now immediate. Given accuracy slack $\zeta$, we set $\epsilon = \frac{1}{2} \zeta \ln C$, which splits the global accuracy budget evenly across local regions, so that local deviations do not accumulate beyond the target slack. SALT then checks whether the labeled set induces complete certified coverage. If every point in $D$ is in $B(x_i, \delta_{x_i}(\epsilon, \theta))$ for some $x_i \in S$, then the model meets the accuracy target, and the HITL loop stops.

**Practical concerns.** The guarantee is a sufficient condition based on bounded spectral norms and Lipschitz continuity of $\eta(\cdot)$. These should be viewed as modeling assumptions for certification, not properties that always hold in practice. When local geometry is irregular or labels change sharply, the bound may be conservative and require smaller $\epsilon$ to achieve the same slack. This said, it remains useful when representations are smooth and local behavior is stable.

The Lipschitz constant $\lambda$ is dataset-dependent. In practice, we estimate it via labeled samples: $\lambda = \max_{i \neq j} |\eta(x_i) - \eta(x_j)| / \|x_i - x_j\|$.

---

## 5 Adaptive Sample Selection

The certified coverage framework from Section 4 provides a stopping rule for selective annotation: once the certified local covers induced by the labeled set fully cover $D$, the target labeling accuracy is guaranteed. The remaining challenge is thus algorithmic: how to expand certified coverage with as few annotations as possible.

This section formulates round-wise selective annotation and develops a greedy algorithm with a coverage-adaptive scoring rule that prioritizes samples contributing most to uncovered regions.

### Progressing Towards Certified Coverage

This problem is nontrivial because coverage depends on labels that are unknown before annotation. Once new annotations are obtained, they can change the model's predictions and the coverage geometry, so a batch that appears useful before annotation may no longer be optimal after retraining. Thus, SALT adopts an HITL procedure: at each round, it selects samples using the current model, obtains their labels from human annotators, updates the model with the newly labeled data, and then re-evaluates the certified coverage condition.

Following common practices in active learning, we assume a small labeled set for model initialization and a fixed annotation budget `bgt` in each round. Consider sample selection in round $t$:

- **Input:** Unlabeled $D$, annotated $\tilde{D}_{t-1}$ from previous rounds, model $M_{t-1}$ trained on $\tilde{D}_{t-1}$, tolerance $\zeta$, and round budget `bgt`.
- **Output:** A subset $D_t \subseteq D$ to be annotated in round $t$.
- **Objective:** To maximize the certified coverage attained by the $M_t$ trained on $\tilde{D}_t = \tilde{D}_{t-1} \cup D_t$.

Unfortunately, this round-wise selective annotation problem (RSAP) is intractable, even under the assumption that the predicted labels of all samples in $D_t$ exactly match the eventual human annotations.

**Theorem 5:** RSAP is NP-hard. $\square$

*Proof sketch:* We prove the NP-hardness by reduction from set cover (see [5] for details), which is NP-complete (cf. [29]). $\square$

### Algorithm

Despite this, we develop a greedy algorithm for round-wise sample selection, guided by the certified coverage criterion:

1. It restricts the candidate pool to uncovered points, since points that are already covered are already controlled at the target tolerance and are therefore less likely to improve certified coverage.

2. It adapts selection to the current coverage state. Early rounds should favor points that cover large uncovered regions to quickly recover the dataset's coarse structure, while later rounds focus on harder regions near decision boundaries or high-curvature areas that require finer refinement. This transition should be driven by the coverage state itself, rather than by manually defined stages.

In each round $t$, SALT selects a batch of `bgt` samples to maximize coverage under the current model $M_{t-1}$. Each candidate is scored by its expected contribution to certified coverage. The score favors samples whose coverage regions include many uncovered points, and gradually emphasizes harder regions where predictive quality is lower as coverage increases.

**Algorithm 1: SALT: sample selection at round $t$.**

```
Input: Unlabeled dataset D, annotated samples D̃ₜ₋₁ from previous rounds,
       current model Mₜ₋₁, tolerance ζ, round budget bgt.
Output: Selected samples Dₜ.

/* Compute current coverage and identify uncovered, candidate samples. */
1. Compute Bᵢ ← B(xᵢ, δₓᵢ(½ζ ln C, Mₜ₋₁)) for each xᵢ in D̃ₜ₋₁;
2. Candidate samples ψ₀ ← D \ ⋃ Bᵢ; ψ ← ψ₀; Dₜ ← ∅;

3. for k = 1 to bgt do
4.     x⁽ᵏ⁾ ← arg max_{x∈ψ} score(x, ψ);  /* Greedy selection by scoring. */
5.     B⁽ᵏ⁾ ← B(x⁽ᵏ⁾, δₓ₍ₖ₎(½ζ ln C, Mₜ₋₁));  /* Get pseudo coverage. */
6.     ψ ← ψ \ B⁽ᵏ⁾;  /* Update candidate samples. */

7. return Dₜ ← {x⁽¹⁾, x⁽²⁾, ..., x⁽ᵇᵍᵗ⁾};
```

Algorithm 1 shows the selection process at round $t$, constructing $D_t$ via greedy coverage expansion. It first evaluates coverage under $M_{t-1}$: for each labeled sample $x_i$, it computes the radius $\delta_{x_i}$ and coverage ball $B_i$ (line 1). The uncovered set defines the candidate $\psi$ (line 2), so selection focuses on regions where coverage is lacking.

The algorithm then selects `bgt` samples greedily from $\psi$. At each step, it picks $x^{(k)}$ with the highest score (line 4), balancing coverage gain and boundary refinement. The gain is estimated using a pseudo coverage ball $B(x)$, computed by treating $M_{t-1}(x)$ as a proxy label (line 5). After selecting $x^{(k)}$, all points covered by its pseudo region are removed from $\psi$ (line 6), avoiding redundancy. After `bgt` steps, the selected set $D_t$ is returned for annotation (line 7).

### Coverage-adaptive Scoring

The greedy selection relies on `score(·)`, a scoring function that ranks samples $x$ in the candidate pool $\psi$. Following the design above, it balances global coverage expansion with predictive-quality improvement when the global coverage level increases. Formally, we define the function as:

$$score(x, \psi) = \ln |\psi \cap \mathbb{B}(x)| + \alpha\left(1 - \frac{|\psi|}{|D|}\right) \ln \|\eta'(x) - p(x)\|_2 \tag{5}$$

where $x \in \psi$ is a candidate sample, $\mathbb{B}(x)$ is the pseudo coverage ball centered at $x$ with radius $\delta_x(\frac{1}{2}\zeta \ln C, M_{t-1})$, $\eta'(x)$ is the one-hot vector corresponding to $\arg\max p(x)$, and $\alpha$ is a hyperparameter.

The first term, $\ln |\psi \cap \mathbb{B}(x)|$, estimates the potential coverage gain of selecting $x$. It measures how many candidates would fall inside the pseudo coverage region if $x$ were annotated consistently with $M_{t-1}$. Maximizing this term favors samples that can expand the covered region as much as possible.

The second term, $\ln \|\eta'(x) - p(x)\|_2$, captures the model's predictive quality on $x$. Larger values indicate that the prediction is farther from a confident one-hot distribution, which means the current model is of lower quality at that point and such points often lie near decision boundaries.

The importance of the quality term is modulated by $1 - \frac{|\psi|}{|D|}$, i.e., the current coverage ratio. When coverage is low, this factor is small, so the score is dominated by the coverage term, favoring rapid global expansion. As coverage increases, the factor grows, and the score gradually emphasizes predictive quality.

### Properties

Our selection strategy has the following properties.

#### Bounded Approximation

Under the optimistic assumption that the predicted labels of selected $D_t$ match their human annotations, Algorithm 1 is a $(1 - \frac{1}{e})$-approximation algorithm for RSAP.

**Theorem 6:** Given candidate samples $\psi_0$ at round $t$, define the expanded coverage of a subset $A$ of $\psi_0$ as $f(A) := |\bigcup_{x \in A} (\psi_0 \cap \mathbb{B}(x))|$. Then we have $f(D_t) \geq (1 - \frac{1}{e}) \max_{A \subseteq D, |A| \leq \text{bgt}} f(A)$. $\square$

*Proof sketch:* Greedy selection with `score(·)` maximizes the marginal gain of $f(\cdot)$ when $\alpha(1 - \frac{|\psi|}{|D|}) = 0$. It thus coincides with the classical greedy algorithm for set cover on universe $\psi_0$, with collection of sets $\{\psi_0 \cap \mathbb{B}(x) : x \in \psi_0\}$ and $|A| \leq \text{bgt}$. Since $f(\cdot)$ is monotone and submodular, the standard result for greedy coverage [50] provides the constant approximation ratio $1 - \frac{1}{e}$. $\square$

#### Adaptive Selection Focus

The strategy adapts to the current coverage level. In the early stages, when coverage is low, $\alpha(1 - \frac{|\psi|}{|D|}) \approx 0$, so the first term dominates. Consequently, the greedy strategy reduces to classical coverage maximization and inherits its approximation guarantee, prioritizing samples with large coverage gain from dense uncovered regions that rapidly recover the dataset's coarse structure.

In later stages, as coverage increases, the second term in Equation 5 becomes increasingly important, and the selection focus naturally shifts from expansion to refinement. At that point, the remaining candidates tend to concentrate near decision boundaries or in high-curvature regions. Annotating such hard points improves the model after retraining, which can further enlarge certified coverage in subsequent rounds.

**Example 3:** Continuing Example 2, Figure 4 shows the adaptive behavior of SALT: newly selected samples are highlighted as centers of red balls. Starting from 4 initial samples in Round 1, SALT first spreads annotations across dense uncovered regions, capturing the dataset's coarse structure. In Round 2, selection shifts toward the remaining dense areas in the two left clusters, beginning to probe harder, high-curvature regions. By Round 3, it concentrates near the decision boundaries (black dotted lines), refining class separation with higher precision. It progresses naturally based on the coverage mechanism, without manually specified stage transitions. $\square$

#### Beyond Diversity or Uncertainty

Equation 5 unifies diversity and uncertainty through the same geometric quantity, namely, $\|\eta' - p\|_2$. A smaller discrepancy yields larger coverage, while a larger one signals low predictive quality and proximity to decision boundaries. Thus, coverage expansion and quality improvement are not separate heuristics, but two views of the same geometry.

#### Complexity

Let $n = |D|$, $m = |\tilde{D}_{t-1}|$. At each round, computing coverage balls for the $m$ labeled samples takes $O(m)$ time. Each of the `bgt` greedy steps then scores all candidates in $\psi$, yielding a worst-case time of $O(m + \text{bgt} \cdot n^2)$. In practice, the cost decreases across rounds as $\psi$ shrinks with increasing coverage.

#### Hyper-parameter Tuning

The scoring function introduces $\alpha$ to balance coverage gain and predictive quality (Equation 5). Its value is dataset-dependent, as the two terms may differ in scale. In practice, we set $\alpha$ so the two terms have comparable influence at the start of selection, preventing either from dominating early, while allowing the adaptive factor $1 - \frac{|\psi|}{|D|}$ to gradually shift the focus from coverage expansion to boundary refinement (see Section 7).

---

## 6 Integrating Weak Supervision

This section shows how SALT integrates weak supervision to reduce human cost when labeled data is scarce. It introduces a logic-based correction mechanism that leverages noisy labeling functions while controlling their errors, and supports effective early-stage learning without compromising the certified accuracy guarantee.

### 6.1 Weak Supervision with Labeling Functions

We first present weak supervision via labeling functions, which provide scalable but noisy labels to accelerate early-stage learning.

Weak supervision [8, 55, 58] derives approximate labels from labeling functions (LFs), such as heuristics or external tools. For example, sentiment analysis may match keywords such as "excellent" to identify positive texts. In dataset labeling, LFs provide auxiliary supervision that is especially valuable when human labels are scarce: they expand limited "gold" labels in early rounds, speed up convergence, and reduce annotation cost.

Formally, let $D = \{x_i\}_{i=1}^n$ be an unlabeled dataset. Weak supervision produces a matrix $\Lambda \in \mathbb{W}^{n \times m}$ over $m$ available LFs, where each $LF_j$ outputs a weak label $\lambda_{ij} \in \mathbb{W} = \{1, ..., C, \perp\}$ for $x_i$. Here $C$ is the number of possible classes, and $\perp$ denotes absence of a label assignment, i.e., $LF_j$ does not assign a label to $x_i$.

#### Caveats

Despite its appeal, weak supervision has inherent limitations, especially under accuracy requirements; thus, it is called "weak". Individual LFs can be inaccurate, and their errors are often hard to detect. As a result, conflicts arise across LFs, mislabels may propagate and distort the learned geometry, and these effects can persist, ultimately limiting the achievable labeling quality.

#### Remedies

Weak labels improve efficiency but, without refinement, can violate the accuracy objective. SALT addresses this with two mechanisms: (1) it introduces logical rules (LRRs) to resolve LF conflicts and filter unreliable signals by combining LF outputs, inter-sample relations, and gold-calibrated soft labels; and (2) the model is trained with both refined weak labels and human annotations, while the weight of weak supervision gradually decreases.

## 6.2 Logic-based Correction for Weak Labels

We now introduce a logic-based correction mechanism, based on LRRs, to resolve conflicts and reduce noise in weak labels.

**Logic-based correction.** We define LRRs, starting from predicates.

**Predicates.** A predicate over dataset \(D\) and its weak labels \(\Lambda\) is:

\[p ::= D(x) \mid T(LF, x, w) \mid \neg T(LF, x, w) \mid M(x, x') \mid Pr(x, w) \otimes c\]

where \(D(x)\) declares that \(x\) is a sample of \(D\), \(w \in \mathbb{W} = \{1, \dots, C, \perp\}\) is the calibrated probability of having label, \(\otimes \in \{>, <, \geq, \leq\}\), and \(c \in [0, 1]\) is a threshold. We elaborate the predicate types as follows.

**(1) Tool predicate \(T(LF, x, w)\).** It returns true if \(LF(x) = w\), making use of the output of an LF. LFs may include heuristics such as regular expressions or keywords for text classification, and metadata filters or pattern matchers for images [55]. More advanced LFs may also use frequency, co-occurrence or pattern order. More generally, SALT can treat outputs from existing weak supervision tools as LFs, e.g., WeaSEL [61], FlyingSquid [25] and MeTaL [6].

**(2) Relationship predicate \(M(x, x')\).** It returns true if a model \(M\) determines that \(x\) and \(x'\) satisfy a predefined relation. For text, e.g., \(M_{topic}\) may identify whether two samples share the same topic [73]. For images, \(M_{geo}\) may use metadata to infer whether samples were captured near the same location [13]. Such relations propagate information across samples and can improve weak-label quality.

**(3) Soft label predicate \(Pr(x, w) \otimes c\).** Here \(Pr(x, w)\) is the probability of \(x\) having label \(w\), predicted by a model \(P\) that fuses weak labels \(\Lambda\) and human annotations. The predicate returns true if \(Pr(x, w) \otimes c\) holds, i.e., the probability satisfies the threshold \(c\) under operator \(\otimes\).

The fusion model \(P\) follows the intuition behind Snorkel LabelModel [57]: agreement patterns among noisy LFs reveal source reliability and enable soft-label inference. We extend this with gold-label calibration, using the available human annotations as strong signals so that the estimated reliabilities are anchored by trusted labels rather than weak-label statistics alone.

Concretely, model \(P\) takes as input the weak label vector produced for a sample \(x_i\) by all LFs, i.e., the \(i\)-th row of \(\Lambda\), and outputs a probability \(Pr(x_i, w)\) for each label \(w \in \mathbb{W}\). Internally, \(P\) introduces a latent true label and treats each LF as a noisy source with its own class-conditional error profile, so the posterior is obtained by combining the estimated reliabilities of all LFs.

We train \(P\) with a joint objective over weak and strong labels (see [5]). Weak labels learn LF reliabilities from agreement patterns, while human annotations serve as ground truth for calibration. As \(P\) is more robust than naive voting, downweighting noise and producing calibrated soft labels suitable for LRRs.

**Label Refining Rules.** An LRR \(\varphi\) has the following form:

\[\varphi: X \to x.label = w,\]

where \(X\) is a conjunction of predicates over \(D\) and \(\Lambda\), and \(x\) is a variable ranging over samples in \(D\) (i.e., \(D(x)\) appears in \(X\)). The \(x.label = w\) assigns a refined label \(w \in \mathbb{W}\) to \(x\). We refer to \(X\) as the precondition and \(x.label = w\) as the consequence of \(\varphi\).

An LRR \(\varphi\) is **conflict-resolving** if its consequence assigns a valid class label \(w \in \{1, \dots, C\}\). If \(w = \perp\), then \(\varphi\) is a **filtering rule**, as it blocks unreliable weak labels from propagating downstream.

Intuitively, an LRR encodes when a weak label should be trusted, corrected or filtered based on a structured pattern of signals in \(X\). Tool predicates indicate which LFs are applied, soft-label predicates summarize confidence from the fusion model, and relationship predicates provide supporting or contradicting evidence from related samples. In this sense, an LRR synthesizes prior experience about how these signals combine to produce reliable labels.

LRRs make a robust refinement layer. They (1) confirm a label if sources agree, (2) override weak heuristics with stronger evidence, (3) propagate support through relationships, or (4) filter labels (with \(\perp\)) when signals conflict. By resolving conflicts before training, LRRs yield more stable supervision for downstream learning.

**Example 4:** Below are LRRs learned from the AGNews dataset [34].

**(1)** \(\varphi_1: D(x) \land T(LF_{regex}, x, business) \land T(LF_{keyword}, x, sports) \to x.label = sports\).

This LRR states that a text sample \(x\) should be classified as sports when both business expressions and sports keywords are present, despite the fact that it has some business-specific expressions. It suggests that, in AGNews, sports keywords are more indicative than business terms. This rule is used to resolve conflicts between LFs and produce a more reliable label.

**(2)** \(\varphi_2: D(x) \land T(LF_{stock}, x', business) \land M_{topic}(x, x') \to x.label = business\).

This LRR assigns label business to \(x\) if it shares a topic with \(x'\), which is identified as business due to stock-related signals. It propagates label evidence across related samples, reflecting that topic-consistent texts tend to share labels. It is used to transfer reliable signals and improve labeling consistency across samples.

**(3)** \(\varphi_3: D(x) \land T(LF_{keyword}, x, scitech) \land Pr(x, scitech) < 0.1 \to x.label = \perp\).

This LRR filters the keyword-based label scitech when the fusion model assigns low confidence (\(<0.1\)). It removes unreliable weak labels for \(x\) when supporting evidence is weak. This rule prevents low-confidence signals from propagating into training. \(\square\)

**Semantics.** A valuation \(h\) of an LRR \(\varphi\) over a dataset \(D\) and its weak labels \(\Lambda\) is a mapping that assigns each variable \(x\) of \(\varphi\) to a sample \(h(x)\) in \(D\), and each variable \(w\) to a label in \(\mathbb{W}\).

We say that valuation \(h\) satisfies a predicate \(p\), denoted \(h \vDash p\), if one of the following conditions holds:
- (1) If \(p\) is \(D(x)\), then \(h(x)\) is a sample of \(D\).
- (2) If \(p\) is \(T(LF, x, w)\) (or \(\neg T(LF, x, w)\)), the tool LF assigns (or does not assign) \(w\) to the sample \(h(x)\).
- (3) If \(p = M(x, x')\), the model \(M\) predicts that \(h(x)\) and \(h(x')\) satisfy the target relation.
- (4) If \(p = Pr(x, w) \otimes c\), the fusion model assigns the sample \(h(x)\) a soft label \(w\) with confidence satisfying threshold \(c\).

For precondition \(X\), \(h \vDash X\) iff \(h \vDash p\) for all predicates \(p\) in \(X\). For an LRR \(\varphi = X \to q\), \(h \vDash \varphi\) if \(h \vDash X\) implies \(h \vDash q\). Dataset \(D\) with \(\Lambda\) satisfies \(\varphi\), written \(D \vDash \varphi\), if all valuations \(h\) over \(D\) and \(\Lambda\) satisfy it. For a set \(\Sigma\) of LRRs, we write \(D \vDash \Sigma\) if \(D \vDash \varphi\) for all \(\varphi \in \Sigma\).

**Remarks.** In practice, we maintain an offline set \(\Sigma\) of LRRs for online HITL use, including both expert-specified and mined rules.

**(a) Rule mining.** We mine LRRs offline from prior similar labeling tasks, treating human annotations as ground truth. The mining process follows a levelwise search in the predicate space. While classic anti-monotonicity does not directly hold due to soft predicates and cross-sample relations, we employ conservative and adaptive pruning: we apply loose support thresholds to retain candidate rules, and further filter them using confidence and coverage impact estimated on human-labeled data. Pruning is relaxed when support is low and gets stricter as support for an LRR increases. This preserves important rules and reduces expensive search (see [5] for details).

**(b) Rule conflict handling.** Conflicts may arise when multiple LRRs apply to the same sample with different consequences. Although such cases are rarely encountered in practice, we include a lightweight handling strategy for completeness. For each sample, we rank the applicable rules by an evidence-aware score that combines rule confidence on human-labeled data, rule specificity, and agreement with the fusion model's soft labels. We then select the consequence of the highest-ranking rule. If the evidence is weak or the top candidates remain too close, we abstain and assign \(\perp\) rather than forcing a label. Details are deferred to [5].

The learned rules also admit a consistency analysis for conflict detection, via standard dependency-analysis methods [22–24].

**LRRs in the HITL pipeline.** Weak-label refinement with LRRs integrates naturally into the SALT HITL pipeline, as the core of the Conflict Resolution module (Figure 2). It takes as input the current human annotations, the weak-label matrix \(\Lambda\) from all LFs, and a rule set \(\Sigma\), and produces one refined weak label per sample in \(D\).

Operationally, the fusion model \(P\) is first updated using the current human annotations. Rule predicates are then evaluated using the updated soft labels from \(P\), LF outputs in \(\Lambda\), and relationship-model outputs. When a sample matches an LRR, its consequence gives us a refined label, which is passed to the model-training stage.

These refined labels provide additional supervision, especially in early rounds when human annotations are scarce. By filtering noise and consolidating reliable signals, LRRs support more effective bootstrapping, improve the initial model quality, and reduce the number of samples that require human annotation.

**Training with weak and strong signals.** In each HITL round, after obtaining refined weak labels and human annotations, we train the labeling model \(M_\theta\) by combining both. Weak labels provide scalable but noisy supervision, while human annotations are reliable but limited. Thus, training should leverage weak labels for early bootstrapping without letting their noise dominate.

We adopt weighted training [11, 44]. Let \(D_W\) be the refined weakly labeled set and \(D_S\) be the human-labeled set, we optimize:

\[\mathcal{L}(t) = e^{-\rho t} \sum_{(x, \tilde{y}) \in D_W} \ell(x, \tilde{y}) + (1 - e^{-\rho t}) \sum_{(x, y) \in D_S} \delta_x \ell(x, y)\]

as the epoch-dependent objective, where \(t\) is the training epoch, \(\delta_x\) is the local cover radius of \(x\), and \(\rho > 0\) is the decay rate. The factor \(e^{-\rho t}\) directly controls the weight of weak supervision at epoch \(t\), decaying over time to shift training toward the more reliable \(D_S\).

**Complexity.** The training is dominated by forward and backward passes of \(M_\theta\) over \(D_W\) and \(D_S\). For \(T\) epochs, the total cost is \(O(T(|D_W| + |D_S|) \cdot C_M)\), where \(C_M\) is the per-sample cost of one optimization step. Thus, adding refined weak labels does not change the training complexity; it just increases the size of the training set.

## 7 EXPERIMENTAL STUDY

Using real-life datasets, we evaluated SALT for its (a) labeling accuracy, (b) human annotation cost, and (c) efficiency compared with active learning. We also empirically tested its sensitivity to hyperparameters and its effectiveness in weak-label integration.

### Experimental settings

We start with the settings.

**Table 1: Summary of classification datasets.**

| Dataset | Type | #Samples | #Classes | #Init / bgt | α |
| --- | --- | --- | --- | --- | --- |
| THUCNews | Chinese news articles (~907 chars) | 68.6k | 14 | 10/40 | 3.0 |
| DBPedia | Wiki articles (~48 words) | 70k | 14 | 10/30 | 3.0 |
| RESISC45 | Remote sensing images | 31.5k | 45 | - | - |
| CIFAR100 | 32×32 color images | 50k | 100 | - | - |

**Datasets.** We evaluated SALT on four real-world classification datasets, as summarized in Table 1.
- (a) **THUCNews** [30] consists of Chinese news articles, with an average length of 907 characters, each labeled with one of 14 classes.
- (b) **DBPedia** [45] contains 14 categories of Wikipedia articles.
- (c) **RESISC45** [16] is a remote sensing image scene classification dataset, covering 45 scene classes with 700 images per class.
- (d) **CIFAR100** [43] contains 32 × 32 color images in 100 categories, with 500 images per category.

We used the training split for all datasets, with ground-truth labels built-in. For selective annotation, we treat each dataset as unlabeled, and reveal a label only when the sample is selected for annotation. That is, we simulate human annotators with a hypothetical oracle that returns the true label on request. For each dataset, we randomly sampled a small initial labeled set (Table 1), and used the same set for all methods. This setup is standard in active learning [4, 10].

**Labeling models.** For text datasets, we used Qwen text embedding v4 [77] to compute a 768-dimensional embedding for each document. For image datasets, we used DINOv2 [52] to extract a 768-dimensional feature vector from each image. We then fed these embeddings into a four-layer MLP for classification. In each round, the models of all methods were reinitialized randomly [4], i.e., we used cold-start rather than warm starting from the previous model [3].

**Baselines.** We tested SALT against SOTA active learning methods:
- **Uncertainty-based:**
  - (1) **Margin** [62], selecting samples with the smallest gap between the top two predictions;
  - (2) **Entropy** [74], selecting samples with the highest predictive entropy.
- **Diversity-based:**
  - (3) **Coreset** [64], which prioritizes samples farthest from the labeled set;
  - (4) **TypiClust** [41], which selects samples difficult to distinguish from labeled ones.
- **Hybrid solutions:**
  - (5) **BADGE** [4], a \(k\)-means++ method based on gradient embeddings w.r.t. the model's output weights;
  - (6) **UHerding** [10], which favors samples farthest from labeled ones with uncertainty regularization;
  - (7) **ALFAMix** [54], which selects samples via feature-space interpolation.

We further include:
- (8) **SALT<sub>noWS</sub>**, which disables weak-label integration in SALT;
- (9) **SALT<sub>cov</sub>** and (10) **SALT<sub>unc</sub>**, which remove the uncertainty and the coverage terms, respectively, in greedy scoring.

To test the noise control of SALT in weak-label integration, we also evaluated the SOTA weak supervision methods, including Snorkel [55], FlyingSquid [25], majority voting and random.

**SALT configurations.** For coverage-adaptive scoring, we set the default round budget `bgt` and the hyperparameter \(\alpha\) separately for each dataset, as listed in Table 1. We also set the decay rate to \(\rho = 0.5\) for weighted training with integrated weak labels and selected human annotations. The Lipschitz coefficient \(\lambda\) for each dataset is estimated per round.

For text classification datasets, we used a set of curated LFs from WRENCH [7] and a few keyword-based LFs generated by ChatGPT. Since there are no high-quality LFs available for RESISC45 and CIFAR100, we disabled weak-label integration in those cases.

To integrate weak labels over THUCNews and DBPedia, we used 21 and 16 LRRs, respectively (see [5] for representative LRRs).

**Testbed.** Our testbed is a server powered by 2× Intel Xeon Gold 6254 @3.10GHz CPUs, each with 16 physical cores and hyperthreading, and 64 GB DDR4 RAM. It also has an NVIDIA Tesla V100 32GB GPU. The operating system is Ubuntu 22.04.2 LTS with Linux kernel version 5.15.0, CUDA 12.2, Python 3.10.19 with PyTorch 2.91. Each experiment is repeated five times, and the average is reported here.

### Experimental results

We next report our findings. To verify that SALT addresses the core challenges outlined in Section 1, we evaluate the certified accuracy (Exp-1), the human annotation cost (Exp-2), and the noise control in weak-label integration (Exp-3). We further study the overall efficiency of the SALT pipeline (Exp-4), its sensitivity to hyperparameters (Exp-5), and ablation studies (Exp-6).

![Figure 5: Accuracy, certified accuracy gap, labeling efficiency, and execution time of SALT.](https://aka.doubaocdn.com/s/VLJc1wfD5H)

**Exp-1: Accuracy.** We compare labeling accuracy of SALT against active learning baselines. We find the following.

**Overall labeling accuracy.** Figures 5a–5d report labeling accuracy across datasets. As annotations increase, SALT is consistently the most accurate, with larger gains on more challenging datasets.

On the 14-class DBPedia, SALT is 16.7–72.4 points more accurate than active learning in the low-label setting (e.g., < 120 annotations). As more labels are added, the gap narrows to 2.1–6.1 points, but SALT remains the best throughout.

The same pattern appears across datasets. With only 320 annotations over the 14-class THUCNews, SALT reaches 82% accuracy, 6.2–19.5 points higher than active learning without weak supervision; its final accuracy is 1.9–9.2 points higher than the baselines. These results show that coverage-driven selection improves dataset-wide labeling quality, prioritizing samples that are both informative and representative.

The final gap is smaller on both image datasets (0.36–7.8 points), due to their class-balanced distribution and well-separated clusters in the embedding space. Thus, most methods become competitive with sufficient labels. This said, SALT remains up to 31.2 percentage points more accurate with fewer than 300 annotations.

**Certified accuracy.** We also validate the certified accuracy guarantee.

Figures 5e-5f show the risk gap (i.e., \(|R_D(\theta_S) - R_D(\theta_D)|\)) against coverage ratio on DBPedia and CIFAR100. We set \(\zeta = 0.1\) for both datasets, such that the target risk gap (the dashed horizontal line) is \(0.1 \ln C\) by Corollary 4. For SALT, the risk gap falls below the target after 150 and 2000 annotations on DBPedia and CIFAR100, respectively, at coverage ratios of 90.9% and 81.0%. Full coverage is reached later at around 420 and 2500 annotations, confirming Corollary 4: full coverage is sufficient to meet the accuracy target.

In contrast, active learning methods fail to meet the coverage condition efficiently. The best baseline on DBPedia (resp. CIFAR100) is diversity-based TypiClust, which reaches a coverage ratio of 74.4% (resp. 91.2%) with 450 (resp. 2500) annotations, but it has the worst risk gap. Uncertainty and hybrid methods fail to achieve high coverage, due to the lack of support within the dense regions far from the decision boundary. This justifies coverage-driven selection.

Overall, these results empirically validate both the correctness and practical value of the certified accuracy guarantee.

**Sensitivity to local Lipschitzness.** We study the sensitivity of SALT to local Lipschitzness on CIFAR100 after 7 annotation rounds.

Figure 5g shows that most samples lie in relatively flat regions with small local Lipschitz constants. The heatmap further illustrates how the local cover radius varies with Lipschitzness. As expected, the two are strongly correlated: regions with larger local Lipschitz constants are steeper and therefore admit smaller local cover radii; in contrast, flatter regions admit larger ones.

Figure 5h shows a different trend. The local accuracy gap within each coverage ball has little correlation with Lipschitzness. This supports our theory: while local curvature affects the size of coverage balls, their accuracy guarantees remain stable across regions.

**Exp-2: Human annotation cost.** SALT's higher labeling accuracy translates into lower human cost. Figures 5a-5d mark the target tolerance \(\zeta = 0.1\). Across all datasets, SALT requires 25.9–75.7% fewer annotations than active learning baselines to reach this accuracy.

Figures 5i-5k further examine the annotation cost under varying accuracy tolerance \(\zeta\). SALT consistently reduces human effort. For instance, at \(\zeta = 0.1\), on DBPedia, THUCNews and CIFAR100, SALT requires only ~64, ~146 and ~973 annotations, while active learning baselines use 3.26–4.11×, >2.50× and 1.35–3.01× more, respectively. Under a looser accuracy tolerance, the gap is even larger: at \(\zeta = 0.15\), SALT is expected to reduce human cost by 75.3–83.9% on DBPedia and THUCNews. This validates the coverage-adaptive greedy strategy, which prioritizes samples that are both informative and representative, thus improving annotation efficiency.

The performance gap is smaller on CIFAR100 than on text datasets. This is because the much larger annotation budget on CIFAR100 allows all methods to better correct early selection bias and converge over time. Moreover, its class-balanced distribution (i.e., 500 samples per class) enables recovery from suboptimal early selections. In contrast, DBPedia has greater class overlap and a more skewed distribution, making sample selection quality critical throughout the annotation process; thus SALT is more effective.

For all methods, human cost increases rapidly as \(\zeta\) decreases. This is expected and reflects diminishing returns in selective annotation: once easy and well-covered regions are labeled, further gains rely on hard boundary cases, which require denser human supervision.

![Figure 6: Weak label integration of SALT.](https://aka.doubaocdn.com/s/fZnR1wfD5H)

**Exp-3: Weak label integration.** To evaluate noise control in isolation, we decouple the weak-label integration component from the HITL pipeline and compare SALT against representative weak supervision methods on weak-label accuracy as a micro-benchmark (Figures 6a-6b).

On THUCNews and DBPedia, SALT covers 31.6k and 51.6k samples, yielding 11.0% and 3.2% lower coverage than weak supervision, respectively. This reduction is expected: unlike existing methods that aggregate most available weak signals, SALT uses filtering LRRs to reject labels when evidence is insufficient, removing low-confidence labels before downstream model training.

Despite covering fewer samples, SALT produces more higher-quality weak labels, with accuracy 2.2–5.9 and 3.6–7.8 points higher than weak supervision on THUCNews and DBPedia, respectively. These results confirm the effectiveness of LRRs: rather than simply combining LF outputs, LRRs reconcile supportive signals and suppress unreliable ones, retaining a smaller but much cleaner label set that is better suited for bootstrapping model training.

**Exp-4: Efficiency.** Figures 5l compares SALT with active learning methods in terms of execution time per round on DBPedia. We exclude model training, as it is a fixed overhead shared by all methods under the same budget. The results are consistent on other datasets.

**(1)** In the early rounds, SALT spends more time on selecting samples, because its greedy selection procedure has \(O(m + bgt \cdot n^2)\) complexity with \(n\) remaining candidates. However, this overhead gets smaller rapidly as the HITL process proceeds. Once more samples are covered, the candidate pool becomes much smaller, and the round cost drops accordingly. For example, the execution time of round 2 (coverage ratio = 77.0%) is only 5.5% of that of round 1 on DBPedia, which is consistent with our theoretical analysis.

**(2)** To reach \(\zeta = 0.1\), SALT takes 62.5%, 58.7%, and 44.0% less time than UHerding, KCGreedy and ALFAMix, respectively. It is magnitudes faster than BADGE and TypiClust. This improvement comes from two sources. First, as discussed above, the cost of each SALT round decreases over time. Second, and more importantly, SALT reaches the same target accuracy in much fewer rounds than the baselines. Hence, its higher selection overhead in early rounds is amortized by faster convergence of the overall HITL workflow.

**(3)** Margin and Entropy use cheaper selection heuristics and are 0.04–0.37s faster than SALT per round. However, reaching the same target accuracy requires much more annotations. As human labeling dominates HITL cost, their savings are outweighed by the additional annotation effort, making SALT more efficient overall.

![Figure 7: Hyperparameter sensitivity and ablation study of SALT.](https://aka.doubaocdn.com/s/2zKa1wfD5M)

**Exp-5: Hyperparameter settings.** We study the sensitivity of SALT to its main hyperparameters, including the round budget `bgt`, the weight \(\alpha\) in coverage-adaptive scoring, and the decay rate \(\rho\).

**Impact of `bgt`.** Figure 7a shows that, given the same total annotation budget, a larger `bgt` usually improves early accuracy. This is because early SALT rounds prioritize coverage expansion, so larger batches cover the embedding space more effectively and provide a stronger initial training signal. As annotations accumulate, the accuracy differences across `bgt` values diminish. In practice, it is preferable to use a larger `bgt` in early rounds and a relatively smaller one later, so as to obtain faster feedback on the accuracy certificate.

**Impact of \(\alpha\).** Figure 7b shows the final labeling accuracy of SALT is robust to different uncertainty weights \(\alpha\) in our selection heuristic: on DBPedia (resp. CIFAR100), accuracy stays within ±0.8% (resp. ±0.1%) of the best setting when \(\alpha \in [1, 5]\) (resp. [0.3, 0.7]). Outside this range, it degrades in both directions: a small \(\alpha\) overemphasizes coverage and underexplores boundary regions, while a large \(\alpha\) overfocuses on hard but redundant samples, reducing annotation efficiency. In practice, we set \(\alpha\) by a brief validation sweep and pick the value that best balances annotation-accuracy tradeoff.

**Impact of \(\rho\).** Figure 7c varies the decay rate \(\rho\) in the weighted training objective and reports final labeling accuracy after 10 rounds. Accuracy levels off when \(\rho \geq 0.5\), but drops for smaller values. A small \(\rho\) slows the decay of weak-label weights, allowing noisy signals to persist and bias training. We recommend \(\rho = 0.5\), which balances early bootstrapping with reliable convergence.

**Exp-6: Ablation study.** Finally, we study the impact of key components in the SALT pipeline, by testing their ablations.

**Impact of weak signals.** Figures 5a–5b show that incorporating LFs allows SALT to converge faster than SALT<sub>noWS</sub>, with the largest gains in the low-label setting (< 120 and < 60 annotations, respectively). On THUCNews, SALT is 5.2% more accurate, and on DBPedia by up to 6.4%. After 8 rounds of annotation, the gap between SALT and SALT<sub>noWS</sub> is reduced to within ±0.1%, a negligible difference since human annotations become dominant in training.

**Impact of the coverage term in sample selection.** Figure 7d compares SALT with SALT<sub>unc</sub>, its variant without coverage (i.e., pure uncertainty sampling), on CIFAR100. SALT consistently beats SALT<sub>unc</sub> in labeling accuracy. SALT<sub>unc</sub> yields low and unstable coverage, reaching only 53.2% even with 2.5k annotations. It thus reduces to uncertainty active learning, making the certified stopping rule much less practical. This shows that the coverage term is necessary to effectively connect certified coverage to actual labeling accuracy.

**Impact of the uncertainty term in sample selection.** Figure 7d also includes SALT<sub>cov</sub>, which removes the uncertainty term. As shown there, SALT<sub>cov</sub> matches SALT in early rounds but falls behind later, showing a 0.75–0.95% accuracy deficit after more than 1k annotations. Although SALT<sub>cov</sub> attains a higher coverage ratio, removing the uncertainty term prevents it from concentrating on hard boundary regions once the model becomes better calibrated.

Together, the two ablations highlight the complementary roles: coverage helps translate the theoretical certificate into actual accuracy, and uncertainty enables targeted refinement in later stages.

### Summary

SALT addresses the key challenges of selective annotation with both theoretical guarantees and empirical results. We find the following.

**(1) Accuracy.** SALT has the highest labeling accuracy across text and image datasets, outperforming active learning baselines by 25.9–72.4 percentage points.

**(2) Annotation efficiency.** It reduces human annotations by up to 75.7% under high-accuracy targets (\(\zeta \leq 0.1\)).

**(3) Weak supervision.** LRR-based refinement yields cleaner weak labels, improving their accuracy by up to 7.8 points.

**(4) End-to-end efficiency.** Although early rounds incur higher selection cost, faster convergence amortizes this overhead, reducing total execution time by >44.0%.

**(5) Robustness.** SALT is robust to hyperparameters: larger `bgt` improves accuracy; accuracy remains consistent for \(\rho \geq 0.5\) (recommended \(\rho = 0.5\)); performance is stable for small variations in \(\alpha\), which should be set per dataset.

**(6) Ablation.** Coverage and uncertainty are complementary: coverage drives early exploration, while uncertainty refines later stages. LRR-filtered weak supervision is essential for efficient bootstrapping without compromising the certified accuracy guarantee.

