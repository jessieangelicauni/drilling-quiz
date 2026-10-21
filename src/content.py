# -*- coding: utf-8 -*-
"""Full text of the KO-DRILL paper. Separated from the DOCX builder."""

TITLE = ("KO-DRILL: A Psychometrically Grounded Large Language Model Pipeline "
         "for Personalised Korean Drill Quizzes")

AUTHORS = "Jessie Angelica"
AFFIL = ("Independent Researcher\n"
         "Artificial Intelligence and Language Learning Technologies\n"
         "jessieangelica21@gmail.com")

ABSTRACT = (
    "Drill practice—short, repeated question-and-answer exercises—is among the most "
    "reliable ways to consolidate the grammar and vocabulary of a second language, but it "
    "only works when each question is pitched at the right level for the individual learner. "
    "Large language models (LLMs) can now write such questions on demand, which removes the "
    "long-standing bottleneck of authoring exercises by hand. It also creates a new and "
    "largely unexamined problem. When questions are produced continuously, every question is "
    "new, so the system has no record of how difficult it actually is. Classical adaptive "
    "testing assumes a fixed, pre-tested question bank; deep knowledge tracing assumes dense "
    "histories in which many learners answer the same questions. In a generative setting "
    "neither assumption holds. This paper presents KO-DRILL, a pipeline for personalised "
    "Korean drill quizzes built around that difficulty. We make three design arguments and "
    "support them with evidence. First, deep learning should be used selectively: an LLM for "
    "writing Korean, a small fine-tuned encoder for predicting difficulty from text, and an "
    "explicit psychometric model—not a neural network—for tracking what a learner knows. "
    "Second, the Elo rating system, the default choice in adaptive practice platforms, is "
    "specifically unsuited here; in a simulation of 120,000 responses we reproduce its "
    "documented failure, with the spread of learner ratings inflating to 2.3 against a true "
    "value of 1.0 and still rising, while the Urnings algorithm, with its acceptance step "
    "corrected for adaptive item selection, remains calibrated at 1.0. "
    "Third, we propose misconception-conditioned distractors. An LLM writes each item so "
    "that every wrong option realises one named misconception, checked without asking a "
    "model to judge: a KAGAS-style annotator confirms each distractor is a minimal edit of "
    "the key in the tested category, and log-probability scoring under a second model "
    "confirms the key is the only correct answer. A multinomial choice model, updated "
    "online, then reads which misconception a learner holds from the option they pick. In "
    "simulation it names the dominant misconception for 58% of learners after ten wrong "
    "answers (chance 17%), and after a misconception is remediated finds the new one for "
    "53%, against 40% for a full-history fit. The next item is then personalised in which "
    "error it confronts, not only in how hard it is. "
    "We validate the novelty with targeted comparisons rather than a broad ablation grid, and set out an exploratory classroom pilot as secondary evidence."
)

INDEX_TERMS = ("Adaptive learning, computerised adaptive testing, educational technology, "
               "item calibration, Korean language learning, large language models, "
               "personalised learning, psychometrics, question generation, Urnings algorithm.")

# ---------------------------------------------------------------- body
BODY = [

("H1", "I. Introduction"),

("P", "Learning a language is, to an unglamorous degree, a matter of repetition. Learners "
      "need to meet the same grammatical pattern many times, in varied contexts, spaced out "
      "over weeks, and they need to be told promptly when they get it wrong. This is the "
      "principle behind drill practice, and it is well supported: retrieval practice, in "
      "which the learner must produce an answer rather than merely re-read material, "
      "produces more durable memory than passive review, and scheduling that practice "
      "according to a model of forgetting improves retention further still [26]."),

("P", "Korean presents a particularly strong case for drill practice, and a particularly "
      "awkward one for existing tools. Korean is an agglutinative language: grammatical "
      "information is carried by particles attached to nouns and by endings attached to "
      "verbs. A learner must choose correctly among markers that have no direct equivalent "
      "in English, decide between the topic marker and the subject marker in contexts where "
      "both are grammatical but convey different emphasis, conjugate verbs that follow "
      "irregular patterns, and select an appropriate level of politeness for the social "
      "relationship being described. Corpus studies of learner writing confirm that these "
      "are precisely the areas where errors concentrate and persist [22], [23]. They are "
      "also areas that reward exactly the kind of high-volume, finely targeted practice "
      "that is tedious for a human teacher to prepare."),

("P", "Until recently the limiting factor was supply. Producing a few thousand well-formed "
      "practice items, each targeting one grammatical point at a known level of difficulty, "
      "with plausible wrong answers, is weeks of expert labour. Large language models (LLMs) "
      "have changed this: systematic reviews now document a large and rapidly maturing "
      "literature on automatic question generation, with models producing multiple-choice "
      "items at a quality that is beginning to approach human-authored material [2], [3], "
      "[4]. The bottleneck has moved."),

("P", "Where it has moved to is the subject of this paper. A personalised drill system must "
      "answer two questions on every turn: how good is this learner, and how hard is this "
      "question? Adaptive testing has mature machinery for both, but that machinery assumes "
      "a fixed bank of questions that have been administered to a calibration sample before "
      "being used in earnest. A generative system violates this assumption completely. If "
      "items are written on demand, every item is, at the moment it is first served, an "
      "item about which nothing is known. One might hope the LLM could simply be asked how "
      "hard its own question is. The evidence says otherwise: LLM self-estimates of item "
      "difficulty are not a reliable substitute for calibration from real responses: "
      "simulated-respondent studies find a single LLM reproduces human item parameters only "
      "partially—correlated at the item level, but with a narrower ability distribution and "
      "less response variability than real learners, improving only when several models are "
      "pooled [8]—and small encoders fine-tuned specifically for difficulty regression "
      "outperform far larger general-purpose models at the task outright [7]. One "
      "might instead reach for deep knowledge tracing, the dominant neural approach to "
      "learner modelling [17], [18], [19]. But knowledge tracing learns from many learners "
      "answering the same items; with an item pool that is effectively infinite and a long "
      "tail of items seen a handful of times, there is nothing for it to learn from. This "
      "is the cold-start problem in an unusually severe form, affecting not a minority of "
      "items but, in principle, all of them [31]."),

("P", "This paper takes that problem as its starting point and makes four contributions."),

("L", "A critical design analysis, in Section III, of which parts of this problem actually "
      "call for deep learning and which do not—an argument that the field's default of "
      "end-to-end neural modelling is the wrong instinct here."),
("L", "A simulation, also in Section III, showing concretely that the Elo rating system—the "
      "most widely deployed tracker in adaptive practice platforms—fails in exactly the "
      "regime a generative item pool creates, quantifying how badly, and showing that "
      "Urnings avoids the failure only when its update is corrected for adaptive selection."),
("L", "One architectural novelty, set out in Section IV: misconception-conditioned "
      "distractors, generated by an LLM and verified by annotation and scoring rather than "
      "by a model's judgement, with an online choice model that infers which misconception "
      "a learner holds from the distractor they choose and steers the next item towards it."),
("L", "A pre-registered experimental design, in Section V, capable of distinguishing the "
      "contribution of each component rather than only evaluating the system as a whole."),

("P", "Throughout, we have tried to keep the prose readable by people who build language "
      "learning products as well as by people who publish on psychometrics, and we gloss "
      "technical terms on first use."),

("H1", "II. Related Work"),

("H2", "A. Automatic Item Generation with Language Models"),

("P", "Automatic item generation—having software write test questions—long predates LLMs, "
      "but transformer-based models changed its practical reach. A 2025 systematic review "
      "of transformer and LLM approaches to multiple-choice question generation finds "
      "consistent improvement over earlier template and rule-based pipelines, while noting "
      "that evaluation in the field remains dominated by surface metrics of fluency rather "
      "than measures of pedagogical usefulness [2]. A broader 2026 survey reaches a similar "
      "conclusion and identifies difficulty control as an open problem [3]. Applied work "
      "confirms feasibility: Badenes-Olmedo and Eyzaguirre's "
      "QuerIA system couples generation with evaluation in a university setting [4]. Dong "
      "and Li address difficulty control directly, combining active learning for labelling "
      "with low-rank adaptation to steer a model towards a requested difficulty level [5]."),

("P", "Distractors—the plausible wrong answers in a multiple-choice item—have received "
      "separate attention, because they are harder to produce than the correct answer and "
      "matter more than is often appreciated. A wrong answer that nobody would ever choose "
      "contributes nothing; a wrong answer that is in fact also correct actively damages the "
      "test. Bitew et al. show that predictive prompting improves distractor plausibility "
      "over prior approaches [6], but the reported ceiling—roughly half of generated "
      "distractors rated high quality by teachers—indicates how much room remains. Crucially, "
      "the entire line of work optimises for plausibility. None of it asks the question we "
      "think is more useful: what does the choice of a particular wrong answer tell us about "
      "the learner? A parallel line of work uses LLMs not to generate educational content but "
      "to judge it, as Li et al. do for student-generated material in learnersourcing [10]; "
      "this is closer to the verification role we describe in Section IV-B, though it has not "
      "been applied to checking the grammatical validity of generated distractors."),

("H2", "B. Estimating How Difficult an Item Is"),

("P", "If a generated item is to be used adaptively, its difficulty must be estimated before "
      "any learner has answered it. Two strands of recent work bear on this. The first asks "
      "whether LLMs can judge difficulty directly, and the answer is qualified rather than "
      "simply negative. Liu et al. simulate respondents with LLMs and calibrate item "
      "parameters from the resulting responses: a single model's item-level parameters "
      "correlate well with human-calibrated ones (Spearman above 0.8 for the strongest "
      "model tested), but no single model reproduces the full human ability distribution—"
      "each LLM's simulated population is narrower and less variable than real test-takers—"
      "so only a pooled ensemble of several models approaches that distribution, and "
      "augmenting real human data with resampled LLM responses helps calibration only with "
      "caveats [8]. The practical reading for a single-generator pipeline like ours is "
      "narrower than \"LLMs can self-rate difficulty\": a single LLM is not a population of "
      "learners, and nothing in this result licenses asking one model to introspect about "
      "an item's difficulty on its own."),

("P", "The second strand is more encouraging and, for our purposes, decisive. Li et al. "
      "treat difficulty estimation as a supervised regression problem over item text and "
      "fine-tune language models to solve it [7]. Their central finding is that small "
      "encoder models—BERT and RoBERTa, with hundreds of millions of parameters rather than "
      "tens of billions—outperform much larger models on this task, achieving a root mean "
      "squared error of 0.2926 and beating the winning entry of a shared-task competition. "
      "Larger is not better here. The task is a narrow mapping from text to a scalar, with "
      "limited labelled data; a compact encoder fine-tuned on exactly that mapping is better "
      "matched to it than a general-purpose generative model prompted to introspect. This "
      "result directly supports the architecture we propose in Section IV."),

("H2", "C. Tracking What a Learner Knows"),

("P", "Two traditions address learner modelling, and they are further apart than they should "
      "be. The deep learning tradition—knowledge tracing—treats the learner's history as a "
      "sequence and trains a neural network to predict the next response. Recent work in "
      "this line is sophisticated: Jiang et al. incorporate item difficulty into a "
      "multi-view attention architecture [18]; Qian et al. ground an attention mechanism in "
      "cognitive theory [19]; Tong and Ren combine knowledge tracing with cognitive load "
      "estimation to generate learning paths, reporting 87.5% prediction accuracy and a "
      "24.6% improvement in learning efficiency [17]. Cognitive diagnosis models pursue "
      "interpretability within the same broad family, with NeuralCD imposing a monotonicity "
      "constraint so that diagnostic output remains meaningful [20] and later work adding "
      "multidimensional features [21]."),

("P", "A related strand models the learner as a persistent profile rather than a response "
      "sequence: Aydın and Yurdugül construct a \u201clearner digital twin\u201d from accumulated "
      "experience and use it to select instructional scaffolding [30]. This shares our "
      "motivation of maintaining an explicit, inspectable learner representation, though it "
      "does not address item calibration."),

("P", "These models are powerful, and they share a structural prerequisite: a fixed set of "
      "items or knowledge components, each observed many times across many learners. "
      "Practical evaluations have found that knowledge tracing models degrade on new "
      "learners, struggle to predict incorrect responses, and can be sensitive to item "
      "ordering in ways that are hard to justify pedagogically. In a generative setting the "
      "prerequisite simply fails."),

("P", "The psychometric tradition takes the opposite approach: a small number of explicit, "
      "interpretable parameters updated by a transparent rule. The Elo rating system, "
      "borrowed from chess, is the workhorse. Each learner and each item carries a single "
      "number; after every response both numbers move towards or away from each other in "
      "proportion to how surprising the outcome was. It is trivial to implement, needs no "
      "training data, and updates in constant time, which is why it underpins deployed "
      "adaptive platforms [16]. Even inside such deployments, though, plain Elo keeps "
      "needing patches: a recent audit of a large-scale deployed system finds item drift "
      "and an unmodelled speed–accuracy tradeoff [14], and separate work finds its fixed "
      "update size forces a tradeoff between tracking real ability change and rating "
      "volatility, proposing a dynamically adjusted update size instead [15]. Section III-B "
      "explains why, despite these virtues, it is the wrong choice for the system we are "
      "describing."),

("P", "The Urnings algorithm offers an alternative with the same operational simplicity and "
      "better statistical behaviour [12]. Each learner and item is represented not by a "
      "point estimate but by an urn containing a fixed number of balls, some green and some "
      "red; the proportion of green balls is the current estimate and the number of balls "
      "controls its precision. Updates exchange balls between the learner's urn and the "
      "item's urn, with a Metropolis–Hastings correction that keeps the procedure "
      "statistically honest. Because the urn has a fixed size, the estimate comes with a "
      "known standard error—something Elo cannot provide—and the algorithm explicitly "
      "corrects for the distortion introduced when items are chosen adaptively. It extends "
      "naturally to tracking several abilities at once [12], and recent work augments it "
      "with response times [13]."),

("H2", "D. Korean Language Technology and Learner Language"),

("P", "Korean-specific resources have matured considerably. Song et al. substantially enrich "
      "the Korean learner corpus for grammatical error correction and writing assessment "
      "[22], building on earlier error-annotated collections; Chun and Kim analyse "
      "substitution errors between topic and nominative markers in a learner corpus and "
      "derive pedagogical applications from them [23]. On the modelling side, Yu et al.'s "
      "KRongBERT demonstrates that explicitly accounting for Korean morphology in a "
      "pre-trained encoder improves downstream performance [24], a reminder that Korean's "
      "agglutinative structure is not something a general multilingual model handles for "
      "free."),

("P", "Generative AI applied specifically to Korean language pedagogy is thin next to the "
      "NLP resources above: the broader second-language-learning literature discusses the "
      "redistribution of agency between learner, teacher and machine that generative tools "
      "bring about [1], but structured, difficulty-calibrated assessment is not its focus. "
      "Assessment-focused studies, meanwhile, concentrate on scoring learner production "
      "rather than generating practice: automated essay scoring with LLMs has been examined "
      "for validity and reliability with mixed conclusions [28], [29]."),

("H2", "E. Lessons from Large-Scale Deployed Systems"),

("P", "Before proposing new methods it is worth asking what the most successful deployed "
      "practice systems already do, since reviewers familiar with Duolingo or similar "
      "products will ask the same question. The honest answer is that there is no single "
      "industry standard — four of the most-cited systems solve the problem four different "
      "ways. Duolingo's practice app schedules review using Half-Life Regression [26], a "
      "lightweight model trained on thirteen million learning traces, since evolved into "
      "Birdbrain, an LSTM-based system reported to process tens of billions of predictions "
      "daily across the product's user base [35]. The Duolingo English Test, a separate "
      "system from the practice app, instead scores test-takers with classical item "
      "response theory inside a computerised-adaptive-testing framework [36]. ALEKS, a "
      "widely deployed mathematics tutor, uses neither: it is built on knowledge space "
      "theory, a combinatorial framework that represents a learner's state as the precise "
      "subset of skills mastered and selects items that move the learner to an adjacent "
      "reachable state [37]. Math Garden uses the Elo-based design discussed in Section "
      "III-B [38]. Anki's default scheduler, FSRS, fits a compact per-card memory model "
      "from review history [39]."),

("P", "Two things are notable about this list. First, none of these four systems trains a "
      "large neural network on the scale of Birdbrain unless it has Birdbrain's data volume "
      "to support it; ALEKS and the original Math Garden design both predate the deep "
      "learning era and remain in wide use, which is itself evidence that a lighter model is "
      "not a compromise but frequently the better engineering choice. Even FSRS, parametric "
      "but far smaller than a neural tracker, is reported to show its advantage over the "
      "classical SM-2 heuristic only once a learner has accumulated a few thousand reviews; "
      "below that volume the two are reported to perform comparably. Our target population "
      "of 50 to 180 learners sits far below the data regime in which any of the heavier "
      "methods were shown to pay off, which is a second, independent argument for the "
      "model-size position taken in Section III-A."),

("P", "Second, and more important for motivating this paper: in every one of these systems, "
      "the exercise content is authored and quality-assured by people, at an organisational "
      "scale an academic project cannot match, before it ever reaches the adaptive engine. "
      "Birdbrain selects among existing exercises; it does not write new ones. The Duolingo "
      "English Test's items are pre-calibrated before deployment, which is precisely the "
      "precondition classical computerised adaptive testing requires. ALEKS's knowledge "
      "space and Math Garden's item bank were both built by sustained human effort over "
      "years. None of these systems has had to solve the problem this paper addresses — an "
      "item with no prior calibration data, served the moment it is written — because none "
      "of them generates items this way. Public reporting on Duolingo’s own "
      "experiments with LLM-written lesson material describes it as an active, "
      "carefully monitored pilot rather than a component of the production "
      "scheduling engine, consistent with the pattern above: even the best-resourced "
      "player in this space has not folded on-demand generation into the system "
      "that Birdbrain itself runs."),

("P", "This reframes the contribution of the present paper. It is not an attempt to out-"
      "engineer Birdbrain, which would be a poor use of effort given the data gap just "
      "described. It is an attempt to reproduce, for a research team without a professional "
      "content-authoring organisation, the adaptive quality that large-scale systems achieve "
      "by combining editorial labour with classical or lightly-parameterised statistical "
      "models. We therefore follow industry practice exactly where our data regime matches "
      "theirs — classical or lightweight trackers over learner ability, and a decay-based "
      "scheduler over review timing, in the spirit of [26] and [39] — and depart from it "
      "only where our resource constraint, not our technical ambition, forces a different "
      "answer: generating and calibrating items we cannot afford to commission."),

("H2", "F. Research Gaps"),

("P", "Reading these literatures together, including the deployed systems just discussed, "
      "three gaps stand out. They are not independent; each follows from the one before, and "
      "the last is precisely the gap that large-scale systems sidestep by not generating "
      "items at all."),

("P", "G1. Wrong answers are wasted. Distractors are optimised to be plausible [6], not to "
      "be informative. When a learner selects a particular wrong answer they have revealed "
      "something quite specific about their internal model of the grammar, and current "
      "systems discard that signal, recording only a binary outcome. Cognitive diagnosis "
      "models would make use of it [20], [21], but they require a pre-specified mapping from "
      "items to skills that generated items do not come with."),

("P", "G2. Rating systems are used outside their validity conditions. Elo is deployed in "
      "adaptive platforms [16] in the one configuration where its measurement properties "
      "break down—simultaneous adaptive item selection and dynamic item calibration—and this "
      "has only recently been characterised precisely [11]. A generative item pool forces "
      "exactly that configuration, because items must be calibrated as they are used. We "
      "are not aware of any language learning system that uses Urnings."),

("P", "G3. Feedback is generated uniformly, and its benefits are not. Thomas et al., studying "
      "on-demand LLM explanatory feedback across seven lessons with 885 learners, found "
      "statistically significant benefits in only two of them, with effect sizes of 0.28 and "
      "0.33 after adjusting for learners' propensity to request feedback [27]. Feedback is "
      "not uniformly valuable, yet systems generate it uniformly, at uniform cost. No "
      "existing work allocates generative effort according to how much the system stands to "
      "learn from the interaction."),

("H1", "III. Design Rationale: Choosing the Right Model Class"),

("P", "This section addresses three design questions that we think are usually answered by "
      "default rather than by argument. Because the answers determine the architecture in "
      "Section IV, we set them out explicitly."),

("H2", "A. Is Deep Learning the Right Tool Here?"),

("P", "Partly. Treating 'personalised Korean quizzing' as one problem to be solved by one "
      "neural system would be a mistake. It decomposes into three sub-problems with quite "
      "different characteristics, and the right answer differs for each."),

("P", "Generating Korean text. Here deep learning is not merely appropriate but "
      "unavoidable. Producing natural, varied, contextually appropriate Korean sentences "
      "that target a specified grammatical point is beyond template systems, which produce "
      "stilted and repetitive output that learners quickly learn to game. An LLM is the "
      "right instrument. We note, though, that raw fluency is no longer the binding "
      "constraint—modern models all produce fluent Korean. The binding constraint is "
      "control: generating an item that targets exactly one grammatical point, at "
      "approximately a specified difficulty, with distractors that are wrong in a specified "
      "way. That is an architectural problem, not a question of which model is most fluent."),

("P", "Predicting difficulty from item text. Deep learning is appropriate, but the "
      "instrument should be small. This is a supervised regression from a short text to a "
      "scalar, with training data measured in thousands rather than millions of examples. "
      "Li et al.'s finding that fine-tuned BERT-class encoders outperform far larger models "
      "on precisely this task [7] is not an anomaly; it reflects the match between a "
      "compact discriminative model and a narrow mapping. A small encoder is also cheap "
      "enough to run on every generated item in real time, deterministic, and "
      "straightforward to recalibrate on local response data as it accumulates."),

("P", "Tracking what the learner knows. Here we argue deep learning is the wrong choice, "
      "and this is the least conventional of our positions. Three reasons. First, data "
      "economy: knowledge tracing models need large interaction logs, and a generative item "
      "pool guarantees that most items are seen by very few learners, so the dense "
      "item-by-learner matrix these models consume never materialises. Second, "
      "interpretability and auditability: a system that reports a learner's proficiency in a "
      "way that must align with an external standard such as TOPIK needs estimates that can "
      "be explained and defended, not a hidden state vector. Third, and most importantly, "
      "uncertainty: the selection policy needs to know not just its estimate of a learner's "
      "ability but how confident it is, and neural trackers do not natively provide "
      "calibrated standard errors. An explicit psychometric model provides all three "
      "properties at a tiny fraction of the computational cost."),

("P", "The resulting position is a division of labour rather than a rejection of deep "
      "learning: neural models generate and perceive; an explicit statistical model "
      "remembers and decides. Deep learning serves the psychometric model rather than "
      "replacing it."),

("H2", "B. Is the Elo Rating System the Right Choice?"),

("P", "No—and this is the clearest-cut of the three questions. Elo is the default in "
      "adaptive practice platforms [16] for good reasons: one number per learner, one per "
      "item, constant-time updates, no training phase. But Bolsinova et al. have recently "
      "characterised its failure mode with precision, and the configuration in which it "
      "fails is exactly the one a generative drill system is forced into [11]."),

("P", "The mechanism is a feedback loop. Elo is well behaved when items are selected at "
      "random, and tolerable when item difficulties are held fixed. It misbehaves when "
      "items are selected adaptively on the basis of current ratings and item difficulties "
      "are themselves being updated. The reason is that selection then depends on the "
      "current estimation error: a learner whose rating is too high is preferentially served "
      "items whose difficulty is also overestimated, which reinforces rather than corrects "
      "the error. The spread of ratings inflates without converging. A generative item pool "
      "cannot avoid this configuration, because new items must be calibrated while they are "
      "in use; there is no pre-test phase in which to fix their difficulties first."),

("P", "To confirm that this matters at realistic scale rather than only in principle, we "
      "simulated 120,000 responses from 300 learners against a pool of 400 items under a "
      "Rasch generating model, with adaptive selection and simultaneous item updating. "
      "Fig. 4 shows the result. The standard deviation of Elo learner ratings grows "
      "monotonically to 2.32 against a true value of 1.04 (1.71 at 60,000 responses, 2.08 "
      "at 90,000) and is still rising when the simulation ends; root mean squared error "
      "against true ability likewise grows, from a minimum of 0.39 at around 22,000 "
      "responses to 1.35 at the end. The system becomes "
      "less accurate the more data it sees. Notably, rank-order correlation with true "
      "ability remains high at 0.97: Elo continues to order learners correctly while the "
      "scale on which it reports them drifts. A system that only ranks learners might not "
      "notice; a system that selects items by predicted success probability will, because "
      "that prediction depends on the absolute scale."),

("P", "Two remedies are available. Parallel Elo, proposed in [12], maintains "
      "two independent rating chains updated on alternating responses and selects items "
      "using whichever chain is not currently being updated, which breaks the dependence "
      "between selection and current error. In our simulation it reduces terminal spread "
      "from 2.32 to 1.55 and terminal error from 1.35 to 0.59—a clear improvement that does "
      "not fully resolve the problem. The Urnings algorithm [12] performs better still, "
      "holding spread at 1.00 against the true 1.04 and error at 0.41, flat rather than "
      "rising. Urnings also provides what the selection policy actually needs and Elo cannot "
      "give: a calibrated standard error. We adopt Urnings as the tracker, and retain Elo "
      "and Parallel Elo as baselines in Section V so that the contribution of this choice "
      "can be measured rather than asserted."),

("P", "Urnings is only immune to this failure when it is specified completely, and the "
      "complete specification is easy to miss. Its Metropolis–Hastings acceptance step "
      "assumes the item was drawn independently of the current urn states; under adaptive "
      "selection it was not, and the acceptance ratio must be multiplied by the ratio of the "
      "selection probabilities under the proposed and current states [12], [13]. We select "
      "items with the normalised Gaussian kernel of [13] on the logit distance between the "
      "learner's and the item's estimate, which makes that ratio cheap to compute. Leaving "
      "the correction out is not a small loss: in the same simulation, Urnings without it "
      "inflates to a spread of 2.63 and an error of 1.73, worse than plain Elo. An earlier "
      "version of our own implementation also stored the item urn with the opposite sign "
      "convention to the update rule, and that bug happened to mask the missing correction; "
      "both are fixed in the results reported here."),

("P", "One trade-off deserves mention. Because an urn holds a fixed number of balls, its "
      "resolution is quantised, and in our simulation Urnings showed slightly lower "
      "rank correlation (0.92) than Elo (0.97) even while being far better calibrated in "
      "absolute terms. Resolution is controlled by urn size: repeating the simulation with "
      "urns of 16, 32, 64 and 128 balls gives terminal errors of 0.59, 0.35, 0.32 and 0.36 "
      "respectively. The largest urn is worse again because at roughly 400 responses per "
      "learner it has not yet spread out to the true ability distribution (rating SD 0.72 "
      "against 1.04). Urn size is therefore a tunable accuracy–responsiveness trade-off, "
      "since larger urns also adapt more slowly to genuine change in ability."),

("P", "This comparison starts every item from a flat, uninformative prior. Seeding items "
      "from a predicted difficulty instead (Section III-C) does not change the conclusion. "
      "Across simulated encoders whose predictions correlate 0.96, 0.78 and 0.44 with true "
      "difficulty, eight replications each, Urnings' terminal error is 0.42, 0.42 and 0.68 "
      "(0.41 from a flat start), against 1.41, 1.47 and 1.70 for Elo (1.34) and 0.63, 0.66 "
      "and 0.90 for Parallel Elo (0.60). A poor prior still costs Urnings accuracy; it is not "
      "immune to one. Sizing each item's urn by the encoder's reported uncertainty, so that "
      "an uncertain prediction is overwritten sooner, limits the cost for the weakest encoder "
      "from 0.68 to 0.59, which is why the encoder must report a calibrated variance and not "
      "only a mean."),

("P", "A further argument against Elo is dimensional. Elo assigns one number per learner, "
      "but Korean proficiency is not one thing. A learner may handle particles well and "
      "honorifics poorly. Collapsing this into a scalar discards the structure that makes "
      "personalisation worthwhile. Urnings extends naturally to tracking many abilities "
      "simultaneously [12], which is what the diagnostic scheme in Section IV-B requires."),

("H2", "C. Which Models for Which Role?"),

("P", "The role decomposition in Section III-A implies different models for different jobs, "
      "and we largely endorse a three-way split, with one refinement."),

("P", "Korean generation: EXAONE. LG AI Research's EXAONE family [32] is trained with "
      "Korean as a first-class language rather than as one of many, which matters in two "
      "concrete ways. Its tokeniser segments Hangul efficiently, reducing both cost and the "
      "morphological fragmentation that inflates sequence length for Korean in "
      "English-centric tokenisers—the same consideration that motivates KRongBERT's "
      "morphological factorisation [24]. EXAONE's role in this system is generation only "
      "(item stems, verb lemmas), and its non-reasoning mode is sufficient for that "
      "narrow task throughout—we do not exercise its reasoning mode anywhere, so we do not "
      "claim its hybrid design as a property this system relies on. The escalation policy "
      "in Section IV-C switches Qwen3 between its modes, not EXAONE. The main caveat on "
      "EXAONE is licensing: it is released under terms restricting commercial use, which is "
      "unproblematic for research but must be disclosed, and which is a reason to retain a "
      "permissively licensed fallback."),

("P", "English explanatory feedback: Qwen3. The user-facing explanation is a cross-lingual "
      "task, not an English one: the model must reason about Korean morphology while writing "
      "in English. This favours a model with strong multilingual grounding, and Qwen3 [33], "
      "pre-trained across 119 languages with a large vocabulary and documented strength on "
      "CJK languages, is a better fit than a model whose multilingual coverage is broad but "
      "shallower on Korean. Gemma-class models are a reasonable alternative where a smaller "
      "footprint or vision input is required; we treat the choice as an ablation rather than "
      "a commitment."),

("P", "Difficulty prediction: KLUE-RoBERTa. We endorse this without reservation as the "
      "target architecture. A Korean encoder pre-trained on the KLUE benchmark suite [34] "
      "and fine-tuned for difficulty regression is the right instrument for the reasons "
      "given in Section III-A and directly supported by Li et al. [7]. Our one refinement "
      "is that the encoder should not emit a point estimate. It should emit a mean and a "
      "variance, because that distribution becomes the prior for the psychometric tracker, "
      "and an item the encoder is unsure about should enter the pool with correspondingly "
      "wide uncertainty rather than with false confidence. This feeds directly into the "
      "adaptive selection policy described in Section IV-C."),

("P", "Stated plainly, not left for a reader to discover in the code: fine-tuning this "
      "encoder needs a Korean item set with real difficulty labels, which does not exist "
      "for this syllabus and cannot be produced by engineering effort alone. The system we "
      "evaluate therefore runs a deterministic placeholder behind the identical (μ̂, σ̂²) "
      "interface—morpheme count and grammatical-complexity tag density from a rule-based "
      "parser, deliberately reported with high σ̂² so it never looks more confident than a "
      "trained model would be—swappable for the trained encoder with no downstream change "
      "once calibration data exists. Section III-B reports how much an "
      "inaccurate prior of this kind costs the tracker."),

("H1", "IV. Proposed Method: KO-DRILL"),

("H2", "A. Overview"),

("P", "Fig. 1 shows the architecture. Four model roles are separated explicitly. An item "
      "foundry generates Korean items and their distractors, passes them through a verifier, "
      "and assigns each a difficulty prior. A multidimensional Urnings tracker maintains, "
      "for each learner, one ability estimate per grammatical skill, each with a known "
      "precision. A selection policy picks the next item. An explainer produces English "
      "feedback. The response returns to the tracker, closing the loop."),

("P", "We write σ(z) = 1/(1 + e^{−z}) for the logistic function. The generating model is the "
      "Rasch model, the simplest standard model in item response theory, under which the "
      "probability that learner p answers item i correctly depends on the difference between "
      "the learner's ability θ_{p} and the item's difficulty β_{i}:"),

("EQ", "P(X_{pi} = 1 | θ_{p}, β_{i}) = σ(θ_{p} − β_{i})\t(1)"),

("H2", "B. Novelty: Misconception-Conditioned Diagnostic Distractors"),

("P", "The novelty addresses gap G1: that wrong answers currently carry no information "
      "beyond being wrong. It has two halves, shown in Fig. 2: items whose every distractor "
      "realises one named misconception, and a learner model that reads which misconception "
      "a learner holds from which distractor they choose."),

("P", "Naming the misconceptions. Each skill has a codebook of the misconceptions learners "
      "hold about it, grounded in the error categories of KAGAS, the Korean automatic "
      "grammatical error annotation scheme [43]. The codebook is written by the generator, "
      "not by us: EXAONE lists the common mistaken rules for the grammar point, each with a "
      "correct sentence and the sentence a learner holding that rule would write. An entry "
      "is kept only if a KAGAS-style annotator, which aligns the two sentences' morphemes "
      "(kiwipiepy) and classifies each change by its grammatical category, finds the "
      "example to be exactly one edit, in the category the skill tests. For closed-class "
      "edits (particles, endings, counters) the edit itself names the misconception, for "
      "example PART:에서→에, after merging phonologically conditioned allomorphs such as "
      "을/를, so the same misconception receives the same code in every item. For "
      "open-class edits, such as one verb for another, the code is the codebook entry. The "
      "codebook is generated once per skill and then frozen, so the labels are data an "
      "instructor can audit, not a by-product hidden inside prompts."),

("P", "Generating items. For each item, EXAONE writes a new sentence using the grammar "
      "point, marks the span that tests it, and writes three wrong versions of that span, "
      "each realising a different codebook misconception. When the learner model already "
      "knows which misconception a learner currently holds for the skill, the request "
      "requires one distractor to realise it: the next item then confronts that error "
      "again, and its outcome measures whether the error is still held. The item is "
      "personalised in which error it tests, not only in how hard it is. No sentence, slot "
      "or distractor comes from a hand-written list."),

("P", "Verifying items. An LLM-written distractor is only useful if it is wrong for the "
      "stated reason and wrong at all. Every check below is made without asking a model to "
      "judge, because asking Qwen3-4B whether a distractor is also acceptable failed in all "
      "three framings we tried (a persistent yes-bias, inconsistency in both directions, "
      "and below-chance pairwise comparison). First, the annotator confirms each distractor "
      "is a minimal pair of the key: exactly one edit, in the skill's category. Second, its "
      "code: the annotator's edit for a closed-class change, which must agree with the "
      "codebook entry the generator named; the named entry otherwise. Third, the three codes "
      "differ, so whichever distractor is chosen is informative. Fourth, the key must be the "
      "only correct answer, checked by scoring rather than judging: under Qwen3-4B, a "
      "different model family from the generator, the sentence with the key must have a "
      "higher teacher-forced log-probability than the sentence with each distractor, as in "
      "minimal-pair evaluation of language models. On the curated minimal pairs of the "
      "earlier template system this ordering held for 51 of 52 unique pairs. Generated "
      "pairs are easier than real errors, so we also scored the 246 learner sentences of "
      "the KoLLA corpus [22] that carry a single one-word replacement against their "
      "corrections: Qwen3-4B preferred the correction in 215 (87.4%, 95% CI 82.7-91.0%). "
      "Larger or Korean-specialised scorers did not do measurably better (Qwen3-8B 85.0%, "
      "Kanana-1.5-8B 85.8%, Gemma-3-12B 88.6%, all intervals overlapping), so the smallest "
      "is kept; it also writes the feedback (Section IV-C). The gate therefore misses about "
      "one real error in eight, one reason the codebook and items are rated by instructors "
      "(Section V-B). It is not "
      "applied to subj_topic, whose correct answer follows a discourse convention that a "
      "language model's distribution does not reflect. An item failing any check is not "
      "discarded: the reasons are returned to EXAONE, which revises the item, up to twice, "
      "before it is dropped."),

("P", "Reading the choice. Following the nested logit view of multiple-choice responses "
      "[44], whether a learner answers correctly is the skill tracker's concern (Section "
      "IV-A). Given a wrong answer, which distractor attracts them is a choice among the "
      "distractors actually offered. Learner p holds a propensity η_{p,m} towards each "
      "misconception m of the skill, and picks distractor d_{j}, realising m_{j}, with the "
      "multinomial-logit (Luce) probability, writing η_{pj} for η_{p,m} at m = m_{j},"),

("EQ", "P(d_{j} | wrong) = exp(η_{pj}) / Σ_{k} exp(η_{pk})\t(2)"),

("P", "where the sum runs over the distractors in the item. After each wrong answer, the "
      "propensities of the offered misconceptions move by one stochastic-gradient step on "
      "the log-likelihood of the observed choice,"),

("EQ", "η_{pk} ← η_{pk} + λ ( 1[k = j] − P(d_{k} | wrong) )\t(3)"),

("P", "with λ = 0.4. This is to the multinomial choice model what the Elo update is to the "
      "Bradley–Terry model, and it has the two properties a drill needs. Availability "
      "cancels: a misconception is only ever compared with the others offered in the same "
      "item, so being offered more often does not by itself make it look more attractive. "
      "And change is followed: a constant step weights recent choices more, so when a "
      "misconception is remediated and another takes over, the estimate moves to it. The "
      "learner's dominant misconception for a skill, argmax_{m} η_{p,m}, is what the next "
      "item is asked to target, and repeated choice of the same misconception is one of "
      "the two triggers for detailed feedback (Section IV-C). Cognitive diagnosis models "
      "would need an expert-supplied matrix mapping items to misconceptions [20], [21]; "
      "here that mapping is produced, and verified, as each item is generated."),

("H2", "C. Application Policies: Item Selection and Escalation"),

("P", "One further design decision deserves explicit defence because it departs from "
      "convention, and we are precise about how much weight its justification actually "
      "carries rather than citing it as settled. Classical adaptive testing selects the "
      "item that maximises information about the learner, which for a Rasch model means "
      "an item the learner has roughly a 50% chance of answering correctly. That is "
      "optimal if the goal is to measure someone efficiently. It is not optimal if the "
      "goal is to help them learn: a 50% success rate is demoralising over a long "
      "practice session. Wilson et al. derive 85% as the optimal training accuracy for "
      "gradient-descent binary classifiers, and are explicit that this is not a universal "
      "constant to target in every setting [25]—we do not claim stronger support from "
      "that result than the authors themselves do. What does transfer is the qualitative "
      "shape of the argument, independently supported in human learning specifically by "
      "the desirable-difficulties literature: moderate, well-calibrated challenge "
      "produces more durable learning than either trivial or overwhelming difficulty "
      "[42]. We treat 85% as a principled starting point motivated by both, not a "
      "value proven optimal for Korean grammar drilling, and the discrimination-index "
      "check below is in part how we verify this starting point is not actively harmful. "
      "We therefore serve item i with probability"),

("EQ", "P(i | p) ∝ exp( −(θ̂_{p} − β̂_{i} − logit 0.85)² / (2 · 0.5²) )\t(4)"),

("P", "over the items targeting the skill, a Gaussian kernel around the 85% point as in "
      "[13]. Sampling rather than taking the closest item matters for the tracker: because "
      "the probability of serving each item is known, the Urnings update can be corrected "
      "for the item having been chosen from the current urn states (Section III-B). "
      "This deliberately sacrifices measurement efficiency for learning efficiency, and we "
      "now put a number on that cost rather than leave the trade-off purely stated. Using the "
      "point-biserial discrimination index—the classical test-theory measure of whether a "
      "correct response on an item actually distinguishes stronger from weaker learners—we "
      "simulate a field-test population (θ ~ N(0,1), 2000 learners, 40 replications per "
      "difficulty) and evaluate it at the item difficulty each policy implies for an "
      "average-ability learner. The maximum-information target (β=0, 50% success) gives "
      "discrimination 0.41; our 85%-target policy (β=θ̂−logit(0.85), the centre of Eq. 4) gives 0.34—"
      "both comfortably in the conventionally \"good\" range (≥0.30), a 16% relative drop "
      "rather than a qualitative one. The trade-off is real, as stated, but the items this "
      "policy serves remain psychometrically sound; this was not previously measured and is "
      "not assumed. Vocabulary scheduling across sessions follows a spacing model in the "
      "tradition of half-life regression [26], with the tracker posterior supplying the "
      "ability term; grammar skills are scheduled differently, stated plainly rather than "
      "left for a reader to assume the same mechanism applies to both. A generated grammar "
      "item is mostly non-repeating, so there is no fixed card for a half-life model to "
      "space—a skill is simply due whenever it is not yet mastered. Mastery itself has no "
      "time-based decay under Urnings, so a mastered skill still needs a periodic low-"
      "priority recheck to stay revocable in practice; we currently use a flat 30-day "
      "interval, an explicit placeholder rather than a derived constant. We tried deriving "
      "one from FSRS's own forgetting-curve math first rather than guessing a different "
      "number: it does not transfer, because FSRS's growing intervals assume each simulated "
      "review is spaced, and feeding it many correct-in-a-row Urnings responses to model "
      "\"mastered\" inflates the derived interval to tens of thousands of days—a different "
      "question (long-term fact retention) than the one we are actually asking (how much "
      "practice evidence constitutes mastery). No Korean-skill-retention dataset exists to "
      "fit a principled interval instead, the same data gap underlying Section III-C's "
      "difficulty-prior placeholder; what is logged in the meantime is every recheck "
      "outcome, so the interval can be tuned from evidence once usage data exists rather "
      "than left an untuned guess indefinitely."),

("P", "A second application policy, uniform-cost feedback, is also worth replacing, though "
      "we do not claim the replacement as a research novelty on its own. Each skill "
      "dimension k of a learner carries an urn whose posterior width w_{k} is a natively "
      "available signal of how evenly matched the learner's recent record has been against "
      "the items they faced, maximal when that record sits near a coin flip. This policy "
      "gates Qwen3's feedback generation only—EXAONE's role is item generation, already "
      "finished before a learner answers, and does not itself branch on escalation. By "
      "default the system takes a routine path: Qwen3's fast non-reasoning mode, short and "
      "cached. It escalates to Qwen3's reasoning mode, producing a contrastive explanation "
      "that names the specific misconception, only when the recent record is ambiguous "
      "enough, or a misconception has persisted enough, to be worth the cost:"),

("EQ", "escalate  ⟺  w_{k} > τ   ∨   consecutive(m_{j}) ≥ 2\t(5)"),

("P", "This targets gap G3—that feedback is generated uniformly although its benefits are "
      "not [27]—economically and pedagogically: reasoning-mode inference is substantially "
      "more expensive, and the finding that LLM feedback produced significant gains in only "
      "two of seven lessons [27] suggests uniform feedback dilutes its own effect. We "
      "deliberately do not claim w_{k} is a calibrated estimate of the learner's absolute "
      "distance from their true ability; we tested that stronger claim directly and it does "
      "not hold at this urn size (a controlled simulation gives corr(w_{k}, true error) = "
      "−0.76, and a regression-calibration check [40], [41] finds credible intervals built "
      "from w_{k} are overconfident by 6–11 percentage points). What this policy spends "
      "computation on is ambiguity in the observable record, not estimation error, which is "
      "why Eq. 5 is reported here as a working application policy—measured to escalate on "
      "76–82% of interactions, saving 18–24% of reasoning-mode calls against an "
      "always-escalate baseline—rather than promoted to a third novelty requiring the "
      "stronger claim."),

("H1", "V. Experimental Design"),

("P", "This design follows directly from Section IV: the novelty gets "
      "targeted pieces of evidence rather than an exhaustive ablation grid, because "
      "a large factorial study answers questions this paper does not ask. The offline "
      "validations in Subsection B are the primary evidence for the paper's "
      "claims and stand on their own; the classroom pilot in Subsection C is a secondary, "
      "exploratory extension that is valuable but not a precondition for the technical "
      "claims being true."),

("H2", "A. Materials and Data"),

("P", "The target curriculum this architecture is designed for is a TOPIK-aligned "
      "syllabus spanning levels I through IV, roughly 120 discrete grammatical points; "
      "what is actually built and evaluated here is 11 skills, "
      "covering 7 of the 7 relevant KAGAS part-of-speech categories. The gap between "
      "120 and 11 is simply unimplemented coverage, not a claim about what has been "
      "tested—every number in Section V-B below is reported at the 11-skill scale, never "
      "projected onto the full syllabus. Misconception codes are grounded in the KAGAS "
      "error categories [43] rather than a project-specific scheme, and each skill's "
      "codebook is frozen and published with the items, so that the diagnostic labels a "
      "generated distractor carries are independently auditable. All generated items entering "
      "any evaluation are reviewed by qualified Korean instructors first; we report, but "
      "do not act on, the verifier's agreement with those reviewers, so that the "
      "verifier's standalone accuracy can be measured on its own terms."),

("H2", "B. Validating the Novelty"),

("P", "Can the learner model find the misconception from choices alone? We simulate it "
      "under the nested logit model the method assumes, with the two complications a real "
      "item pool adds. Each skill has six misconceptions and each item offers three, but "
      "unevenly: the r-th misconception is offered with weight 1/r, because some errors are "
      "easier to realise as a clean distractor than others. Learner p answers correctly "
      "with probability σ(θ_{p} − β_{i}); given a wrong answer, they choose by Eq. 2 with "
      "propensities η_{p,m} ~ N(0, 1) plus 1.5 on one dominant misconception, and the target "
      "is the misconception they are actually most drawn to, argmax_{m} η_{p,m}. In a "
      "stable condition propensities never change; in a remediated condition, after one "
      "phase, the dominant misconception is replaced by another, and accuracy is counted "
      "from the switch. We compare the online model of Eq. 3 with a raw count of choices, "
      "the rate of choices per offer, and the full-history maximum-likelihood fit of the "
      "same choice model (ridge 0.1), over 300 learners and 20 replications."),

("P", "After 10 wrong answers, the online model names the dominant misconception for "
      "58.1% of learners when misconceptions are stable (95% interval over replications "
      "57–59%; chance 16.7%), against 59.4% for the full-history fit, 60.0% for the rate "
      "and 45.9% for the count. After a remediation it names the new one for 53.2% "
      "(52–54%), against 40.1%, 40.2% and 34.6%. After 40 wrong answers the figures are "
      "73.1% stable and 73.7% after remediation for the online model, against 78.0% and "
      "59.3% for the full-history fit. The constant step costs about five points when "
      "misconceptions never change, and gains about fourteen when one is remediated, which "
      "is the change a drill exists to produce. The raw count is worst in both conditions, "
      "because it confuses being chosen often with being offered often."),

("P", "Does the item foundry produce verified items, and how often? validate_item_foundry "
      "requests 20 items per skill across the 11 skills, half of them targeted at a "
      "codebook misconception drawn at random, and reports with Wilson intervals the "
      "share accepted after at most two revisions, the reasons replies were rejected, how "
      "many of each skill's codebook misconceptions the accepted items cover, and, as a "
      "second view that is reported but not used as a gate, whether Qwen3-4B picks the key "
      "by letter under every rotation of the options. These runs need a GPU and have not "
      "yet been made with the item foundry; this paragraph is the protocol, and every "
      "accepted item is saved for the instructor review below."),

("P", "Item quality is rated by three independent Korean language instructors on four "
      "dimensions — grammaticality of the stem, uniqueness of the key, plausibility of "
      "each distractor, and correctness of its misconception label — with agreement reported as "
      "Fleiss' κ. Two measures matter most. First, the rate of items containing more than "
      "one defensible answer, since this is the failure the verifier gate in Section IV-B "
      "exists specifically to catch, and it is the one the general distractor-generation "
      "literature does not report [6]. Second, agreement between the codebook misconception a "
      "distractor was generated and verified to realise and the one an instructor independently "
      "assigns after seeing only the sentence, the distractor and the codebook — this is the direct test of whether "
      "misconception-conditioning actually produces a usable diagnosis, or merely a "
      "plausible wrong answer with a label attached. Neither measure is available yet; "
      "this subsection is the protocol, not a result."),

("H2", "C. Exploratory Classroom Pilot"),

("P", "Fig. 5 summarises the design, offered as secondary evidence of real-world "
      "usefulness rather than as the paper's central claim. We recruit learners of Korean "
      "as a foreign language at CEFR A2–B1 equivalent, block-randomised by placement score "
      "into three arms. Arm A receives a fixed expert-authored item bank in random order. "
      "Arm B receives the same fixed bank with Elo-based adaptive selection, representing "
      "current good practice [16]. Arm C receives the full KO-DRILL system. Arms A and B "
      "share an item bank so that the effect of generation is separable from the effect of "
      "adaptivity. Running this pilot at the full scale implied by Fig. 5 — three arms of "
      "60 — is desirable but not required before the claims in Subsection B can "
      "be reported; a smaller single-cohort pilot comparing only Arm A against Arm C is a "
      "reasonable first step if resources are limited, at the cost of not separating the "
      "generation and adaptivity effects."),

("P", "Participants complete a TOPIK-aligned pre-test, six weeks of practice, an "
      "immediate post-test, and a delayed retention test four weeks later. The primary "
      "outcome is normalised learning gain on the post-test; secondary outcomes are "
      "retention, items-to-mastery, per-skill diagnostic accuracy against instructor "
      "judgement, and completion rate. We note plainly that a sample of 60 per arm gives "
      "80% power to detect a standardised between-arm difference of only d = 0.51 "
      "two-sided, or d = 0.59 Bonferroni-corrected across the three pairwise comparisons — "
      "larger than the 0.28–0.33 range reported for LLM feedback alone [27]. This is a "
      "known limitation of the pilot, not of the paper's technical claims, which is "
      "exactly why Subsection B is designed to stand independently of it."),

("H2", "D. Threats to Validity"),

("P", "Four threats apply regardless of which evaluation is run. Novelty effects may "
      "inflate engagement in any condition using the generative system regardless of "
      "instructional quality, which is why retention at four weeks rather than immediate "
      "engagement is the measure weighted most heavily wherever the pilot is run. "
      "Instructor reviewers cannot be blinded to whether an item was generated, since "
      "generated and authored items differ stylistically, which is why verifier–reviewer "
      "agreement is reported as the primary item-quality measure rather than reviewer "
      "ratings alone. KAGAS was developed from corpora of predominantly non-Indonesian "
      "learners and may transfer imperfectly to Indonesian speakers of Korean specifically, "
      "so error-type distributions should be reported by first-language background rather "
      "than assumed to transfer. EXAONE's licence restricts commercial "
      "deployment; the Qwen3-generator configuration noted in Section III-C should be "
      "treated as the one to validate if deployment beyond research use is ever intended. "
      "Finally, Qwen3's free-form explanatory feedback is automatically corrected for one "
      "specific, previously observed failure (an irregular-conjugation slip) and was "
      "tested clean of two other previously observed failures—answering in the wrong "
      "language, and leaking an in-prompt example's topic into the explanation—on four "
      "trials; it has not been exhaustively audited for other factual errors a free-form "
      "explanation could contain, and we do not claim it has."),

("H1", "VI. Discussion and Conclusion"),

("P", "The argument of this paper is that the move from fixed item banks to generated ones "
      "is not a straightforward improvement but a change of regime, and that several "
      "methods inherited from the fixed-bank era stop working when it happens. Deep "
      "knowledge tracing loses the dense interaction data it depends on. Classical adaptive "
      "testing loses the pre-calibrated bank it assumes. The Elo rating system, used in the "
      "one configuration a generative pool makes unavoidable, drifts without converging—a "
      "result we have reproduced here at realistic scale, with terminal error growing by a "
      "factor of more than three from its minimum while rank-order correlation stayed near "
      "0.97, which is to say while the failure stayed invisible to the diagnostics a "
      "practitioner would most naturally check."),

("P", "The remedies we propose are deliberately modest in their use of deep learning. A "
      "small encoder predicts difficulty well enough to initialise a psychometric prior, "
      "which the tracker then corrects from real responses; an "
      "explicit urn-based model, not a neural network, maintains learner state and supplies "
      "the uncertainty that both item selection and compute allocation depend on; and the "
      "large generative models are confined to the two jobs they are uniquely good at, "
      "writing Korean and explaining it. The resulting system is cheaper than an "
      "end-to-end neural alternative and considerably easier to audit, which for an "
      "educational tool is not a secondary consideration."),

("P", "Two limitations should be stated plainly. The simulation in Section III-B "
      "demonstrates a property of the trackers under a Rasch generating model; it does not "
      "demonstrate that KO-DRILL improves learning, which only the study in Section V can "
      "do. And the misconception-conditioning approach assumes that learner errors fall into "
      "a taxonomy that can be enumerated in advance, which is a better approximation for "
      "Korean particles and verb morphology than it will be for pragmatics or discourse."),

("P", "The architecture is not specific to Korean. Any language with a well-documented "
      "inventory of learner errors and an available language-specific encoder could be "
      "substituted. Korean is a good first case precisely because its agglutinative "
      "morphology makes the error taxonomy unusually crisp, and because the gap between what "
      "learners need to drill and what existing tools provide is unusually wide."),
]

REFERENCES = [
 "R. Godwin-Jones, “Distributed agency in second language learning and teaching through generative AI,” <i>Language Learning &amp; Technology</i>, vol. 28, no. 2, pp. 5–31, Jun. 2024, doi: 10.64152/10125/73570.",
 "H. W. Awalurahman, R. F. Aji, and I. Budi, “Transformer and large language models for automatic multiple-choice question generation: A systematic literature review,” <i>IEEE Access</i>, vol. 13, pp. 127100–127112, 2025, doi: 10.1109/ACCESS.2025.3590423.",
 "R. Oliveira, A. Hernández, M. Garbin, J. Dos Reis, and R. Azevedo, “Automatic question generation with large language models: A survey,” <i>ACM Computing Surveys</i>, vol. 58, no. 15, pp. 1–34, Nov. 2026, doi: 10.1145/3832773.",
 "C. Badenes-Olmedo and P. Eyzaguirre, “QuerIA: Adaptive question generation and evaluation in higher education using large language models and contextual learning,” <i>Expert Systems with Applications</i>, vol. 301, p. 130140, Mar. 2026, doi: 10.1016/j.eswa.2025.130140.",
 "X. Dong and Z. Li, “Efficient unified framework for difficulty-controllable question generation: Large language models enabling active learning labeling and prompt-based low-rank adaptation,” <i>IEEE Access</i>, vol. 14, pp. 76358–76371, 2026, doi: 10.1109/ACCESS.2026.3693277.",
 "S. K. Bitew, J. Deleu, C. Develder, and T. Demeester, “Distractor generation for multiple-choice questions with predictive prompting and large language models,” in <i>Communications in Computer and Information Science</i>, Cham, Switzerland: Springer, 2025, pp. 48–63, doi: 10.1007/978-3-031-74627-7_4.",
 "M. Li, H. Jiao, T. Zhou, N. Zhang, S. Peters, and R. W. Lissitz, “Item difficulty modeling using fine-tuned small and large language models,” <i>Educational and Psychological Measurement</i>, vol. 85, no. 6, pp. 1065–1090, Dec. 2025, doi: 10.1177/00131644251344973.",
 "Y. Liu, S. Bhandari, and Z. A. Pardos, “Leveraging LLM respondents for item evaluation: A psychometric analysis,” <i>British Journal of Educational Technology</i>, vol. 56, no. 3, pp. 1028–1052, May 2025, doi: 10.1111/bjet.13570.",
 "Y. Gao, Y. Ma, Y. Qi, and T. Liu, “Development of a computerized adaptive item bank for the Big Five personality based on large language models,” <i>Assessment</i>, Mar. 2026, doi: 10.1177/10731911261427877.",
 "K. Li, C. Qian, and X. Yang, “Evaluating the quality of student-generated content in learnersourcing: A large language model based approach,” <i>Education and Information Technologies</i>, vol. 30, no. 2, pp. 2331–2360, Feb. 2025, doi: 10.1007/s10639-024-12851-4.",
 "M. Bolsinova, B. Gergely, and M. J. S. Brinkhuis, “Keeping Elo alive: Evaluating and improving measurement properties of learning systems based on Elo ratings,” <i>British Journal of Mathematical and Statistical Psychology</i>, vol. 79, no. 1, pp. 95–110, Feb. 2026, doi: 10.1111/bmsp.12395.",
 "M. Bolsinova, M. J. S. Brinkhuis, A. D. Hofman, and G. Maris, “Tracking a multitude of abilities as they develop,” <i>British Journal of Mathematical and Statistical Psychology</i>, vol. 75, no. 3, pp. 753–778, Nov. 2022, doi: 10.1111/bmsp.12276.",
 "B. Gergely and M. Bolsinova, “Toward psychometric learning analytics: Augmenting the Urnings algorithm with response times,” <i>Journal of Educational Measurement</i>, vol. 63, no. 1, Mar. 2026, doi: 10.1111/jedm.70026.",
 "H. Vermeiren, J. Kruis, M. Bolsinova, H. L. J. van der Maas, and A. D. Hofman, “Psychometrics of an Elo-based large-scale online learning system,” <i>Computers and Education: Artificial Intelligence</i>, vol. 8, p. 100376, Jun. 2025, doi: 10.1016/j.caeai.2025.100376.",
 "H. Vermeiren, A. D. Hofman, M. Bolsinova, H. L. J. van der Maas, and W. Van Den Noortgate, “Balancing stability and flexibility: Investigating a dynamic K value approach for the Elo rating system in adaptive learning environments,” <i>User Modeling and User-Adapted Interaction</i>, vol. 36, no. 1, Art. 4, Dec. 2025, doi: 10.1007/s11257-025-09439-z.",
 "B. Vesin, K. Mangaroska, K. Akhuseyinoglu, and M. Giannakos, “Adaptive assessment and content recommendation in online programming courses: On the use of Elo-rating,” <i>ACM Transactions on Computing Education</i>, vol. 22, no. 3, pp. 1–27, Sep. 2022, doi: 10.1145/3511886.",
 "C. Tong and C. Ren, “Deep knowledge tracing and cognitive load estimation for personalized learning path generation using neural network architecture,” <i>Scientific Reports</i>, vol. 15, no. 1, Jul. 2025, doi: 10.1038/s41598-025-10497-x.",
 "X. Jiang, B. Yang, and W. Liu, “DMAM: Difficulty-enhanced multi-view attention-based model for knowledge tracing,” <i>Knowledge-Based Systems</i>, vol. 324, p. 113736, Aug. 2025, doi: 10.1016/j.knosys.2025.113736.",
 "L. Qian, K. Zheng, L. Wang, and S. Li, “Student state-aware knowledge tracing based on attention mechanism: A cognitive theory view,” <i>Pattern Recognition Letters</i>, vol. 184, pp. 190–196, Aug. 2024, doi: 10.1016/j.patrec.2024.06.009.",
 "F. Wang, Q. Liu, E. Chen, Z. Huang, Y. Yin, S. Wang, and Y. Su, “NeuralCD: A general framework for cognitive diagnosis,” <i>IEEE Transactions on Knowledge and Data Engineering</i>, vol. 35, no. 8, pp. 8312–8327, Aug. 2023, doi: 10.1109/TKDE.2022.3201037.",
 "T. Huang, J. Geng, H. Yang, S. Hu, X. Ou, J. Hu, and Z. Yang, “Interpretable neuro-cognitive diagnostic approach incorporating multidimensional features,” <i>Knowledge-Based Systems</i>, vol. 304, p. 112432, Nov. 2024, doi: 10.1016/j.knosys.2024.112432.",
 "J. Song, K. Lim, and J. Park, “Enriching the Korean learner corpus for grammatical error correction and writing assessment,” <i>Language Resources and Evaluation</i>, vol. 60, no. 1, Mar. 2026, doi: 10.1007/s10579-025-09882-9.",
 "J. Chun and M. H. Kim, “Corpus-informed application based on Korean Learners’ Corpus: Substitution errors of topic and nominative markers,” <i>Asian-Pacific Journal of Second and Foreign Language Education</i>, vol. 6, no. 1, Dec. 2021, doi: 10.1186/s40862-021-00112-7.",
 "H. Yu, Y. Cho, G. Park, and M. Kim, “KRongBERT: Enhanced factorization-based morphological approach for the Korean pretrained language model,” <i>Information Processing &amp; Management</i>, vol. 62, no. 3, p. 104072, May 2025, doi: 10.1016/j.ipm.2025.104072.",
 "R. C. Wilson, A. Shenhav, M. Straccia, and J. D. Cohen, “The eighty five percent rule for optimal learning,” <i>Nature Communications</i>, vol. 10, no. 1, Nov. 2019, doi: 10.1038/s41467-019-12552-4.",
 "B. Settles and B. Meeder, “A trainable spaced repetition model for language learning,” in <i>Proc. 54th Annu. Meeting Assoc. Comput. Linguistics (ACL)</i>, Berlin, Germany, 2016, pp. 1848–1858, doi: 10.18653/v1/P16-1174.",
 "D. R. Thomas, C. Borchers, S. Bhushan, E. Gatz, S. Gupta, and K. R. Koedinger, “LLM-generated feedback supports learning if learners choose to use it,” in <i>Lecture Notes in Computer Science</i>, vol. 16063, Cham, Switzerland: Springer, 2026, doi: 10.1007/978-3-032-03870-8_33.",
 "A. Pack, A. Barrett, and J. Escalante, “Large language models and automated essay scoring of English language learner writing: Insights into validity and reliability,” <i>Computers and Education: Artificial Intelligence</i>, vol. 6, p. 100234, Jun. 2024, doi: 10.1016/j.caeai.2024.100234.",
 "S. Crossley, L. Holmes, and W. Morris, “Assessing the reliability and validity of large language models in automatic essay scoring,” <i>Assessing Writing</i>, vol. 69, p. 101082, Jul. 2026, doi: 10.1016/j.asw.2026.101082.",
 "F. Aydın and H. Yurdugül, “Tutoring model for intelligent tutoring systems: Adaptive instructional scaffoldings based on experiences of the learner digital twin,” <i>Technology, Knowledge and Learning</i>, Oct. 2026, doi: 10.1007/s10758-026-10039-y.",
 "B. Hu, Y. Ma, Z. Liu, and H. Wang, “A social importance and category enhanced cold-start user recommendation system,” <i>Expert Systems with Applications</i>, vol. 277, p. 127130, Jun. 2025, doi: 10.1016/j.eswa.2025.127130.",
 "LG AI Research, “EXAONE 4.0: Unified large language models integrating non-reasoning and reasoning modes,” LG AI Research, Seoul, South Korea, Tech. Rep., 2025.",
 "Qwen Team, “Qwen3 technical report,” Alibaba Group, Hangzhou, China, Tech. Rep., 2025.",
 "S. Park, J. Moon, S. Kim, W. I. Cho, J. Han, J. Park, C. Song, J. Kim, Y. Song, T. Oh, J. Lee, J. Oh, S. Lyu, Y. Jeong, I. Lee, S. Seo, D. Lee, H. Kim, M. Lee, S. Jang, S. Do, S. Kim, K. Lim, J. Lee, K. Park, J. Shin, S. Kim, L. Park, A. Oh, J.-W. Ha, and K. Cho, “KLUE: Korean language understanding evaluation,” in <i>Proc. 35th Conf. Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track</i>, 2021.",
 "Duolingo, Inc., “How Duolingo’s AI learns what you need to learn,” Duolingo Engineering Blog, 2023. [Online]. Available: https://blog.duolingo.com/duolingo-technology-innovations/. [Accessed: 18-Oct-2026].",
 "Duolingo, Inc., “An overview of Duolingo English Test administration and scoring,” Duolingo, Pittsburgh, PA, USA, Tech. Rep., 2024. [Online]. Available: https://duolingo-papers.s3.amazonaws.com/reports/Duolingo_whitepaper_test_scoring_2024_v1.pdf.",
 "E. Cosyn, H. Uzun, C. Doble, and J. Matayoshi, “A practical perspective on knowledge space theory: ALEKS and its data,” <i>Journal of Mathematical Psychology</i>, vol. 101, p. 102512, Apr. 2021, doi: 10.1016/j.jmp.2021.102512.",
 "S. Klinkenberg, M. Straatemeier, and H. L. J. van der Maas, “Computer adaptive practice of maths ability using a new item response model for on the fly ability and difficulty estimation,” <i>Computers &amp; Education</i>, vol. 57, no. 2, pp. 1813–1824, Sep. 2011, doi: 10.1016/j.compedu.2011.02.003.",
 "J. Ye and the Open Spaced Repetition community, “Free Spaced Repetition Scheduler (FSRS),” 2022. [Online]. Available: https://github.com/open-spaced-repetition. [Accessed: 18-Oct-2026].",
 "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in <i>Proc. 34th Int. Conf. Mach. Learn. (ICML)</i>, Sydney, Australia, 2017, pp. 1321–1330.",
 "V. Kuleshov, N. Fenner, and S. Ermon, “Accurate uncertainties for deep learning using calibrated regression,” in <i>Proc. 35th Int. Conf. Mach. Learn. (ICML)</i>, Stockholm, Sweden, 2018, pp. 2801–2809.",
 "E. L. Bjork and R. A. Bjork, “Making things hard on yourself, but in a good way: Creating desirable difficulties to enhance learning,” in <i>Psychology and the Real World: Essays Illustrating Fundamental Contributions to Society</i>, M. A. Gernsbacher, R. W. Pew, L. M. Hough, and J. R. Pomerantz, Eds. New York, NY, USA: Worth Publishers, 2011, pp. 56–64.",
 "S. Yoon, S. Park, G. Kim, J. Cho, K. Park, G. T. Kim, M. Seo, and A. Oh, “Towards standardizing Korean grammatical error correction: Datasets and annotation,” in <i>Proc. 61st Annu. Meeting Assoc. Comput. Linguistics (ACL)</i>, Toronto, Canada, 2023.",
 "Y. Suh and D. M. Bolt, “Nested logit models for multiple-choice item response data,” <i>Psychometrika</i>, vol. 75, no. 3, pp. 454–473, 2010.",
]

FIGURES = [
 ("figures/fig1_architecture.png", 6.3,
  "Fig. 1. The KO-DRILL architecture. Four model roles are kept separate: a Korean generator "
  "(EXAONE), a diagnostic distractor generator, a difficulty-prior encoder (target "
  "architecture KLUE-RoBERTa; a deterministic placeholder in the evaluated system, Section "
  "III-C), and an English explainer (Qwen3). Learner state is held by an explicit "
  "multidimensional Urnings tracker rather than by a neural network, so that every ability "
  "estimate carries a known standard error."),
 ("figures/fig2_distractors.png", 6.4,
  "Fig. 2. Misconception-conditioned distractors (Section IV-B). Each distractor is written "
  "to realise one misconception from the skill's codebook and verified by a KAGAS-style "
  "annotator to be a single edit of the key in the tested category; log-probability scoring "
  "under a second model guards against a distractor that is also correct, and a failed item "
  "is revised from the stated reasons. The option a learner chooses updates their "
  "misconception propensities (Eq. 3), and the dominant one is targeted by the next item."),
 ("figures/fig3_escalation.png", 7.0,
  "Fig. 3. Uncertainty-gated escalation (application policy, Section IV-C). Expensive "
  "reasoning-mode generation and contrastive feedback are spent only where the posterior is "
  "wide or a misconception has persisted, rather than uniformly across all interactions."),
 ("figures/fig4_simulation.png", 7.0,
  "Fig. 4. Motivating simulation, no item-difficulty prior injected: 120,000 responses, 300 "
  "learners, 400 items, Rasch generating model, with adaptive selection and simultaneous item "
  "calibration — the regime a generative item pool forces. (a) Elo's rating spread inflates to "
  "2.32 against a true value of 1.04 and is still rising; Urnings, with the selection-corrected "
  "acceptance step, holds at 1.00. (b) Elo's error against true ability grows after roughly "
  "22,000 responses, so the estimates get "
  "worse as more data arrives. This reproduces, at realistic scale, the failure characterised "
  "analytically by Bolsinova et al. [11]."),
 ("figures/fig5_experiment.png", 7.0,
  "Fig. 5. Human study design. Arms A and B share a fixed expert-authored item bank so that the "
  "effect of generation is separable from the effect of adaptivity; Arm C receives the full "
  "system. Retention at four weeks, rather than immediate post-test performance, is the measure "
  "weighted most heavily, to guard against novelty effects."),
]
