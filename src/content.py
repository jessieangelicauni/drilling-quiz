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
    "documented failure, with the spread of learner ratings inflating to 2.7 against a true "
    "value of 1.0 and still rising, while the Urnings algorithm remains calibrated at 1.1. "
    "Third, we propose two novelties: "
    "misconception-conditioned distractors that turn each wrong answer into a specific "
    "diagnosis rather than an undifferentiated error; and an uncertainty-gated escalation "
    "policy that spends expensive model capacity only where measurement is least certain. "
    "Both personalise below the grain a selection policy alone can reach: diagnosis tailored "
    "to the specific misconception an individual answer reveals, and intervention depth "
    "tailored to how uncertain the system is about that learner on that skill—neither "
    "available to a system built around a fixed, pre-calibrated item bank. "
    "We validate each novelty with a single targeted comparison rather than a broad ablation grid, and set out an exploratory classroom pilot as secondary evidence."
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
      "according to a model of forgetting improves retention further still [28]."),

("P", "Korean presents a particularly strong case for drill practice, and a particularly "
      "awkward one for existing tools. Korean is an agglutinative language: grammatical "
      "information is carried by particles attached to nouns and by endings attached to "
      "verbs. A learner must choose correctly among markers that have no direct equivalent "
      "in English, decide between the topic marker and the subject marker in contexts where "
      "both are grammatical but convey different emphasis, conjugate verbs that follow "
      "irregular patterns, and select an appropriate level of politeness for the social "
      "relationship being described. Corpus studies of learner writing confirm that these "
      "are precisely the areas where errors concentrate and persist [24], [25]. They are "
      "also areas that reward exactly the kind of high-volume, finely targeted practice "
      "that is tedious for a human teacher to prepare."),

("P", "Until recently the limiting factor was supply. Producing a few thousand well-formed "
      "practice items, each targeting one grammatical point at a known level of difficulty, "
      "with plausible wrong answers, is weeks of expert labour. Large language models (LLMs) "
      "have changed this: systematic reviews now document a large and rapidly maturing "
      "literature on automatic question generation, with models producing multiple-choice "
      "items at a quality that is beginning to approach human-authored material [5], [6], "
      "[7], [8]. The bottleneck has moved."),

("P", "Where it has moved to is the subject of this paper. A personalised drill system must "
      "answer two questions on every turn: how good is this learner, and how hard is this "
      "question? Adaptive testing has mature machinery for both, but that machinery assumes "
      "a fixed bank of questions that have been administered to a calibration sample before "
      "being used in earnest. A generative system violates this assumption completely. If "
      "items are written on demand, every item is, at the moment it is first served, an "
      "item about which nothing is known. One might hope the LLM could simply be asked how "
      "hard its own question is. The evidence says otherwise: LLM self-estimates of item "
      "difficulty are poorly calibrated, systematically biased towards judging items easier "
      "than they are, and unreliable at the level of the individual item [11], [12]. One "
      "might instead reach for deep knowledge tracing, the dominant neural approach to "
      "learner modelling [19], [20], [21]. But knowledge tracing learns from many learners "
      "answering the same items; with an item pool that is effectively infinite and a long "
      "tail of items seen a handful of times, there is nothing for it to learn from. This "
      "is the cold-start problem in an unusually severe form, affecting not a minority of "
      "items but, in principle, all of them [33]."),

("P", "This paper takes that problem as its starting point and makes four contributions."),

("L", "A critical design analysis, in Section III, of which parts of this problem actually "
      "call for deep learning and which do not—an argument that the field's default of "
      "end-to-end neural modelling is the wrong instinct here."),
("L", "A simulation, also in Section III, showing concretely that the Elo rating system—the "
      "most widely deployed tracker in adaptive practice platforms—fails in exactly the "
      "regime a generative item pool creates, and quantifying how badly."),
("L", "Two architectural novelties, set out in Section IV, that personalise below the "
      "grain a selection policy alone can reach: diagnosis tailored to the specific "
      "misconception an answer reveals, and intervention cost tailored to per-skill "
      "measurement uncertainty."),
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
      "than measures of pedagogical usefulness [5]. A broader 2026 survey reaches a similar "
      "conclusion and identifies difficulty control as an open problem [6]. Applied work "
      "confirms feasibility across subjects: Ling and Afzaal generate question–answer pairs "
      "for higher education from pre-trained models [7], and Badenes-Olmedo and Eyzaguirre's "
      "QuerIA system couples generation with evaluation in a university setting [8]. Dong "
      "and Li address difficulty control directly, combining active learning for labelling "
      "with low-rank adaptation to steer a model towards a requested difficulty level [9]."),

("P", "Distractors—the plausible wrong answers in a multiple-choice item—have received "
      "separate attention, because they are harder to produce than the correct answer and "
      "matter more than is often appreciated. A wrong answer that nobody would ever choose "
      "contributes nothing; a wrong answer that is in fact also correct actively damages the "
      "test. Bitew et al. show that predictive prompting improves distractor plausibility "
      "over prior approaches [10], but the reported ceiling—roughly half of generated "
      "distractors rated high quality by teachers—indicates how much room remains. Crucially, "
      "the entire line of work optimises for plausibility. None of it asks the question we "
      "think is more useful: what does the choice of a particular wrong answer tell us about "
      "the learner? A parallel line of work uses LLMs not to generate educational content but "
      "to judge it, as Li et al. do for student-generated material in learnersourcing [14]; "
      "this is closer to the verification role we describe in Section IV-B, though it has not "
      "been applied to checking the grammatical validity of generated distractors."),

("H2", "B. Estimating How Difficult an Item Is"),

("P", "If a generated item is to be used adaptively, its difficulty must be estimated before "
      "any learner has answered it. Two strands of recent work bear on this. The first asks "
      "whether LLMs can judge difficulty directly, and the answer is largely negative. Liu "
      "et al., simulating respondents with LLMs and analysing the resulting response "
      "patterns psychometrically, find systematic divergence from human response data [12]. "
      "Independent evaluations of LLM difficulty classification report accuracy well below "
      "what operational use would require, with a consistent bias towards rating items as "
      "easier than they prove to be."),

("P", "The second strand is more encouraging and, for our purposes, decisive. Li et al. "
      "treat difficulty estimation as a supervised regression problem over item text and "
      "fine-tune language models to solve it [11]. Their central finding is that small "
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
      "multi-view attention architecture [20]; Qian et al. ground an attention mechanism in "
      "cognitive theory [21]; Tong and Ren combine knowledge tracing with cognitive load "
      "estimation to generate learning paths, reporting 87.5% prediction accuracy and a "
      "24.6% improvement in learning efficiency [19]. Cognitive diagnosis models pursue "
      "interpretability within the same broad family, with NeuralCD imposing a monotonicity "
      "constraint so that diagnostic output remains meaningful [22] and later work adding "
      "multidimensional features [23]."),

("P", "A related strand models the learner as a persistent profile rather than a response "
      "sequence: Aydın and Yurdugül construct a \u201clearner digital twin\u201d from accumulated "
      "experience and use it to select instructional scaffolding [32]. This shares our "
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
      "adaptive platforms [18]. Section III-B explains why, despite these virtues, it is the "
      "wrong choice for the system we are describing."),

("P", "The Urnings algorithm offers an alternative with the same operational simplicity and "
      "better statistical behaviour [16]. Each learner and item is represented not by a "
      "point estimate but by an urn containing a fixed number of balls, some green and some "
      "red; the proportion of green balls is the current estimate and the number of balls "
      "controls its precision. Updates exchange balls between the learner's urn and the "
      "item's urn, with a Metropolis–Hastings correction that keeps the procedure "
      "statistically honest. Because the urn has a fixed size, the estimate comes with a "
      "known standard error—something Elo cannot provide—and the algorithm explicitly "
      "corrects for the distortion introduced when items are chosen adaptively. It extends "
      "naturally to tracking several abilities at once [16], and recent work augments it "
      "with response times [17]."),

("H2", "D. Korean Language Technology and Learner Language"),

("P", "Korean-specific resources have matured considerably. Song et al. substantially enrich "
      "the Korean learner corpus for grammatical error correction and writing assessment "
      "[24], building on earlier error-annotated collections; Chun and Kim analyse "
      "substitution errors between topic and nominative markers in a learner corpus and "
      "derive pedagogical applications from them [25]. On the modelling side, Yu et al.'s "
      "KRongBERT demonstrates that explicitly accounting for Korean morphology in a "
      "pre-trained encoder improves downstream performance [26], a reminder that Korean's "
      "agglutinative structure is not something a general multilingual model handles for "
      "free."),

("P", "Work on AI in Korean language teaching has so far concentrated on conversation and "
      "motivation rather than on structured assessment. Hong and Yang evaluate a "
      "mobile-assisted blended Korean course and report gains in both proficiency and "
      "affective outcomes [1]. Related work in second language learning more broadly "
      "examines generative AI chatbots for speaking practice [3], vocabulary and reading "
      "development [4], and the redistribution of agency between learner, teacher and "
      "machine that generative tools bring about [2]. Assessment-focused studies, meanwhile, "
      "concentrate on scoring learner production rather than generating practice: automated "
      "essay scoring with LLMs has been examined for validity and reliability with mixed "
      "conclusions [30], [31]."),

("H2", "E. Lessons from Large-Scale Deployed Systems"),

("P", "Before proposing new methods it is worth asking what the most successful deployed "
      "practice systems already do, since reviewers familiar with Duolingo or similar "
      "products will ask the same question. The honest answer is that there is no single "
      "industry standard — four of the most-cited systems solve the problem four different "
      "ways. Duolingo's practice app schedules review using Half-Life Regression [28], a "
      "lightweight model trained on thirteen million learning traces, since evolved into "
      "Birdbrain, an LSTM-based system reported to process tens of billions of predictions "
      "daily across the product's user base [37]. The Duolingo English Test, a separate "
      "system from the practice app, instead scores test-takers with classical item "
      "response theory inside a computerised-adaptive-testing framework [38]. ALEKS, a "
      "widely deployed mathematics tutor, uses neither: it is built on knowledge space "
      "theory, a combinatorial framework that represents a learner's state as the precise "
      "subset of skills mastered and selects items that move the learner to an adjacent "
      "reachable state [39]. Math Garden uses the Elo-based design discussed in Section "
      "III-B [40]. Anki's default scheduler, FSRS, fits a compact per-card memory model "
      "from review history [41]."),

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
      "scheduler over review timing, in the spirit of [28] and [41] — and depart from it "
      "only where our resource constraint, not our technical ambition, forces a different "
      "answer: generating and calibrating items we cannot afford to commission."),

("H2", "F. Research Gaps"),

("P", "Reading these literatures together, including the deployed systems just discussed, "
      "three gaps stand out. They are not independent; each follows from the one before, and "
      "the last is precisely the gap that large-scale systems sidestep by not generating "
      "items at all."),

("P", "G1. Wrong answers are wasted. Distractors are optimised to be plausible [10], not to "
      "be informative. When a learner selects a particular wrong answer they have revealed "
      "something quite specific about their internal model of the grammar, and current "
      "systems discard that signal, recording only a binary outcome. Cognitive diagnosis "
      "models would make use of it [22], [23], but they require a pre-specified mapping from "
      "items to skills that generated items do not come with."),

("P", "G2. Rating systems are used outside their validity conditions. Elo is deployed in "
      "adaptive platforms [18] in the one configuration where its measurement properties "
      "break down—simultaneous adaptive item selection and dynamic item calibration—and this "
      "has only recently been characterised precisely [15]. A generative item pool forces "
      "exactly that configuration, because items must be calibrated as they are used. We "
      "are not aware of any language learning system that uses Urnings."),

("P", "G3. Feedback is generated uniformly, and its benefits are not. Lin et al., studying "
      "on-demand LLM explanatory feedback across seven lessons with 885 learners, found "
      "statistically significant benefits in only two of them, with effect sizes of 0.28 and "
      "0.33 after adjusting for learners' propensity to request feedback [29]. Feedback is "
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
      "on precisely this task [11] is not an anomaly; it reflects the match between a "
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
      "adaptive practice platforms [18] for good reasons: one number per learner, one per "
      "item, constant-time updates, no training phase. But Bolsinova et al. have recently "
      "characterised its failure mode with precision, and the configuration in which it "
      "fails is exactly the one a generative drill system is forced into [15]."),

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
      "monotonically to 2.69 against a true value of 1.04 and is still rising when the "
      "simulation ends; root mean squared error against true ability likewise grows, from a "
      "minimum of 0.44 at around 15,000 responses to 1.70 at the end. The system becomes "
      "less accurate the more data it sees. Notably, rank-order correlation with true "
      "ability remains high at 0.97: Elo continues to order learners correctly while the "
      "scale on which it reports them drifts. A system that only ranks learners might not "
      "notice; a system that selects items by predicted success probability will, because "
      "that prediction depends on the absolute scale."),

("P", "Two remedies are available. Parallel Elo, proposed in [16], maintains "
      "two independent rating chains updated on alternating responses and selects items "
      "using whichever chain is not currently being updated, which breaks the dependence "
      "between selection and current error. In our simulation it reduces terminal spread "
      "from 2.69 to 1.71 and terminal error from 1.70 to 0.73—a clear improvement that does "
      "not fully resolve the problem. The Urnings algorithm [16] performs better still, "
      "holding spread at 1.11 against the true 1.04 and error at 0.43, flat rather than "
      "rising. Urnings also provides what the selection policy actually needs and Elo cannot "
      "give: a calibrated standard error. We adopt Urnings as the tracker, and retain Elo "
      "and Parallel Elo as baselines in Section V so that the contribution of this choice "
      "can be measured rather than asserted."),

("P", "One trade-off deserves mention. Because an urn holds a fixed number of balls, its "
      "resolution is quantised, and in our simulation Urnings showed slightly lower "
      "rank correlation (0.92) than Elo (0.97) even while being far better calibrated in "
      "absolute terms. Resolution is controlled by urn size: repeating the simulation with "
      "urns of 16, 32, 64 and 128 balls gives terminal errors of 0.59, 0.45, 0.35 and 0.30 "
      "respectively. Urn size is therefore a tunable accuracy–responsiveness trade-off, "
      "since larger urns also adapt more slowly to genuine change in ability."),

("P", "A further argument against Elo is dimensional. Elo assigns one number per learner, "
      "but Korean proficiency is not one thing. A learner may handle particles well and "
      "honorifics poorly. Collapsing this into a scalar discards the structure that makes "
      "personalisation worthwhile. Urnings extends naturally to tracking many abilities "
      "simultaneously [16], which is what the diagnostic scheme in Section IV-B requires."),

("H2", "C. Which Models for Which Role?"),

("P", "The role decomposition in Section III-A implies different models for different jobs, "
      "and we largely endorse a three-way split, with one refinement."),

("P", "Korean generation: EXAONE. LG AI Research's EXAONE family [34] is trained with "
      "Korean as a first-class language rather than as one of many, which matters in two "
      "concrete ways. Its tokeniser segments Hangul efficiently, reducing both cost and the "
      "morphological fragmentation that inflates sequence length for Korean in "
      "English-centric tokenisers—the same consideration that motivates KRongBERT's "
      "morphological factorisation [26]. And its hybrid design, offering both a fast "
      "non-reasoning mode and a slower reasoning mode, maps unusually well onto the "
      "escalation policy we describe in Section IV-C. The main caveat is licensing: EXAONE "
      "is released under terms restricting commercial use, which is unproblematic for "
      "research but must be disclosed, and which is a reason to retain a permissively "
      "licensed fallback."),

("P", "English explanatory feedback: Qwen3. The user-facing explanation is a cross-lingual "
      "task, not an English one: the model must reason about Korean morphology while writing "
      "in English. This favours a model with strong multilingual grounding, and Qwen3 [35], "
      "pre-trained across 119 languages with a large vocabulary and documented strength on "
      "CJK languages, is a better fit than a model whose multilingual coverage is broad but "
      "shallower on Korean. Gemma-class models are a reasonable alternative where a smaller "
      "footprint or vision input is required; we treat the choice as an ablation rather than "
      "a commitment."),

("P", "Difficulty prediction: KLUE-RoBERTa. We endorse this without reservation. A Korean "
      "encoder pre-trained on the KLUE benchmark suite [36] and fine-tuned for difficulty "
      "regression is the right instrument for the reasons given in Section III-A and "
      "directly supported by Li et al. [11]. Our one refinement is that the encoder should "
      "not emit a point estimate. It should emit a mean and a variance, because that "
      "distribution becomes the prior for the psychometric tracker, and an item the encoder "
      "is unsure about should enter the pool with correspondingly wide uncertainty rather "
      "than with false confidence. This feeds directly into the adaptive selection policy "
      "described in Section IV-D."),

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

("H2", "B. Novelty 1: Misconception-Conditioned Diagnostic Distractors"),

("P", "The first novelty addresses gap G1: that wrong answers currently carry no "
      "information. Fig. 2 shows the design."),

("P", "Rather than inventing a project-specific error scheme, we ground the taxonomy ℳ in "
      "KAGAS, the 14-category Korean Automatic Grammatical error Annotation System [24], "
      "which already defines and independently validates categories spanning particle "
      "substitution, honorific agreement, and verb-conjugation errors, among others. Using "
      "an external, independently defined taxonomy rather than one we compiled ourselves "
      "matters for auditability: the diagnostic label a distractor carries is not a "
      "judgement call this paper made alone. Rather than asking the generator for plausible "
      "wrong answers, we condition it: for a sampled category m_{j} ∈ ℳ, produce the "
      "distractor that a learner holding exactly that misconception would produce, and tag "
      "the distractor with m_{j}."),

("P", "This inverts the usual relationship between item and skill mapping. Cognitive "
      "diagnosis models require a matrix mapping items to skills, supplied by experts [22], "
      "[23]; here the mapping is a by-product of generation, because the system knows which "
      "misconception it asked for. Diagnosis then becomes almost trivial: a learner who "
      "selects d_{j} has supplied evidence against skill m_{j} specifically, and only that "
      "dimension of their ability vector is updated. A drill item becomes a diagnostic "
      "instrument without any additional expert annotation."),

("P", "This only works if the distractors are trustworthy, which is why the verifier gate is "
      "part of the novelty rather than an implementation detail. It enforces three "
      "conditions. First, a morphological analyser confirms that the distractor actually "
      "realises the intended misconception and not some other error. Second—the most "
      "important check, and the dominant failure mode of LLM-generated distractors—it "
      "confirms that the distractor is not itself a correct answer. Korean makes this "
      "failure easy to commit: many particle substitutions that look wrong in isolation are "
      "acceptable in some reading, so a naive generator will regularly produce items with "
      "two correct options. Third, a round-trip check requires a solver model to still "
      "select the intended key."),

("P", "An item failing any check is not simply discarded and resampled from scratch, which "
      "wastes the useful part of a near-miss generation. Instead it passes to a critic: a "
      "second LLM call, separate from the generator, that is shown the failed check and the "
      "item, and is asked to diagnose why it failed and draft a targeted revision—rephrasing "
      "a distractor that turned out to also be correct, for instance, rather than regenerating "
      "the whole item. The revised item re-enters the verifier gate. This generator–validator–"
      "critic structure costs one extra LLM call on the (expected to be minority) path where "
      "verification fails, in exchange for converging on a valid item faster than blind "
      "resampling does."),

("H2", "C. Novelty 2: Uncertainty-Gated Escalation"),

("P", "The second novelty addresses gap G3: that feedback is generated uniformly although its "
      "benefits are not [29]. Fig. 3 shows the policy."),

("P", "Each skill dimension k of a learner carries an urn whose size determines the width "
      "w_{k} of the posterior—a direct, natively available signal of how evenly matched the "
      "learner's recent record has been against the items they faced on this skill, maximal "
      "when that record sits near a coin flip and smaller the further it drifts toward "
      "consistent success or failure. We deliberately do not claim w_{k} is a calibrated "
      "estimate of absolute distance from the learner's true ability—Section V-C reports a "
      "direct test of that stronger claim, and it does not hold. What the gate below spends "
      "computation on is ambiguity in the recent record, not estimation error, a distinction "
      "worth stating plainly rather than eliding. By default the system takes a routine path: "
      "generation in EXAONE's fast non-reasoning mode and short, cached feedback. It "
      "escalates—to reasoning-mode generation of a minimal-pair probe and a contrastive "
      "English explanation naming the specific misconception—only when the recent record is "
      "ambiguous enough, or persistent enough, to be worth the cost:"),

("EQ", "escalate  ⟺  w_{k} > τ   ∨   consecutive(m_{j}) ≥ 2\t(2)"),

("P", "That is, when the learner's recent performance on this skill has been genuinely mixed "
      "against the items they were matched to, or when the learner has repeated the same "
      "diagnosed misconception and a persistent misunderstanding is therefore indicated. Both "
      "conditions are about ambiguity and persistence in the observable record, not about an "
      "unobservable true-ability gap that no deployed system can check itself against."),

("P", "The argument for this is both economic and pedagogical. Economically, reasoning-mode "
      "inference is substantially more expensive than standard decoding, and spending it "
      "uniformly is waste. Pedagogically, the finding that LLM feedback produced significant "
      "gains in only two of seven lessons [29] suggests that uniform feedback dilutes effect: "
      "much of it goes to learners who did not need it. Routing elaborate explanation towards "
      "persistent, specifically diagnosed misconceptions is a principled way to concentrate "
      "it where it should do most good. We are not aware of prior work that allocates "
      "generative compute in an educational system according to this kind of psychometric "
      "ambiguity signal."),

("H2", "D. Selection Policy: Measurement Versus Learning"),

("P", "One further design decision deserves explicit defence because it departs from "
      "convention. Classical adaptive testing selects the item that maximises information "
      "about the learner, which for a Rasch model means an item the learner has roughly a "
      "50% chance of answering correctly. That is optimal if the goal is to measure someone "
      "efficiently. It is not optimal if the goal is to help them learn: a 50% success rate "
      "is demoralising over a long practice session. Wilson et al. show that for a broad "
      "class of learning systems the rate of improvement is maximised at an accuracy of "
      "about 85% [27], and the pedagogical literature on desirable difficulty points the "
      "same way. We therefore select"),

("EQ", "i* = argmin_{i} | σ(θ̂_{p} − β̂_{i}) − 0.85 |\t(3)"),

("P", "restricted to items targeting the skill with the greatest expected pedagogical return. "
      "This deliberately sacrifices measurement efficiency for learning efficiency, and we "
      "now put a number on that cost rather than leave the trade-off purely stated. Using the "
      "point-biserial discrimination index—the classical test-theory measure of whether a "
      "correct response on an item actually distinguishes stronger from weaker learners—we "
      "simulate a field-test population (θ ~ N(0,1), 2000 learners, 40 replications per "
      "difficulty) and evaluate it at the item difficulty each policy implies for an "
      "average-ability learner. The maximum-information target (β=0, 50% success) gives "
      "discrimination 0.41; our 85%-target policy (β=θ̂−logit(0.15), by Eq. 3) gives 0.34—"
      "both comfortably in the conventionally \"good\" range (≥0.30), an 18% relative drop "
      "rather than a qualitative one. The trade-off is real, as stated, but the items this "
      "policy serves remain psychometrically sound; this was not previously measured and is "
      "not assumed. Scheduling across sessions follows a spacing model "
      "in the tradition of half-life regression [28], with the tracker posterior supplying "
      "the ability term."),

("H1", "V. Experimental Design"),

("P", "This design follows directly from Section IV: each of the two novelties gets a "
      "single, targeted piece of evidence rather than an exhaustive ablation grid, because "
      "a large factorial study answers questions this paper does not ask. The offline "
      "validations in Subsections B through C are the primary evidence for the paper's "
      "claims and stand on their own; the classroom pilot in Subsection D is a secondary, "
      "exploratory extension that is valuable but not a precondition for the technical "
      "claims being true."),

("H2", "A. Materials and Data"),

("P", "Grammatical targets are drawn from a TOPIK-aligned syllabus spanning levels I "
      "through IV, decomposed into roughly 120 discrete grammatical points. The "
      "misconception taxonomy is KAGAS [24], the 14-category Korean grammatical error "
      "taxonomy, rather than a project-specific scheme, so that the diagnostic labels a "
      "generated distractor carries are independently defined and independently "
      "auditable. All generated items entering any evaluation are "
      "reviewed by qualified Korean instructors first; we report, but do not act on, the "
      "verifier's agreement with those reviewers, so that the verifier's standalone "
      "accuracy can be measured on its own terms."),

("H2", "B. Validating Novelty 1: Misconception-Conditioned Distractors"),

("P", "Item quality is rated by three independent Korean language instructors on four "
      "dimensions — grammaticality of the stem, uniqueness of the key, plausibility of "
      "each distractor, and correctness of the KAGAS label — with agreement reported as "
      "Fleiss' κ. Two measures matter most. First, the rate of items containing more than "
      "one defensible answer, since this is the failure the verifier gate in Section IV-B "
      "exists specifically to catch, and it is the one the general distractor-generation "
      "literature does not report [10]. Second, agreement between the KAGAS label a "
      "distractor was generated to realise and the label an instructor independently "
      "assigns after seeing only the distractor — this is the direct test of whether "
      "misconception-conditioning actually produces a usable diagnosis, or merely a "
      "plausible wrong answer with a label attached."),

("H2", "C. Validating Novelty 2: Uncertainty-Gated Escalation"),

("P", "Two questions. First, does w_{k} actually track absolute estimation error "
      "|θ̂_{p} − θ_{p}|, the stronger claim Section IV-C explicitly declines to make? We test "
      "this directly rather than assume it: a single simulated learner per fixed true θ, "
      "run against an exactly-known item pool to equilibrium, removing every population and "
      "item-estimation confound. The result is decisive and negative. At n=32, the urn size "
      "used throughout this system, a learner at an extreme true ability converges to one of "
      "the NARROWEST urns—looking most confident—while carrying the LARGEST true error of "
      "any position tested; a learner near the pool's centre gets the WIDEST urn while "
      "carrying unremarkable error. Across a grid of θ values this gives corr(w_{k}, error) = "
      "−0.76 at n=32, reproduced at n=64 (−0.85), weakening only at n=128 (−0.04). We "
      "quantify this formally with the regression-calibration method of [43] (the continuous "
      "analogue of [42]'s classifier calibration error): at every nominal confidence level "
      "tested, the urn's own credible interval for θ, transformed from the proportion to the "
      "logit scale, contains the true value LESS often than claimed—a nominal 90% interval "
      "contains truth only 76-83% of the time, a nominal 50% interval only 34-43% of the "
      "time, a mean calibration error of 6-11 percentage points depending on population. "
      "w_{k} is systematically overconfident as an absolute uncertainty estimate, not merely "
      "uncorrelated with one, because it is a function of the learner's current position "
      "within the urn's fixed range rather than of accumulated evidence quantity. We take "
      "this as settled for n=32 and report it plainly rather than retrofit a weaker claim "
      "the data does not support: the trigger in Eq. 2 targets ambiguous recent performance, "
      "as Section IV-C now states, and should not be read as targeting poorly-measured "
      "learners in any absolute sense. Second, is the gating worth its complexity regardless "
      "of that distinction: because update() does not depend on which feedback path produced "
      "an interaction, the learner's measured trajectory is IDENTICAL whether a given "
      "interaction was escalated or not—gating changes only what is shown to the learner, "
      "never the tracker's arithmetic—so there is no measurement-precision cost to compare. "
      "What is worth reporting is how much compute gating actually saves: across a simulated "
      "practice history of 120,000 responses from 300 learners, the gate fires on 76-82% of "
      "interactions depending on population mix, saving 18-24% of reasoning-mode calls "
      "against an always-escalate baseline at zero measured cost to tracking."),

("H2", "D. Exploratory Classroom Pilot"),

("P", "Fig. 5 summarises the design, offered as secondary evidence of real-world "
      "usefulness rather than as the paper's central claim. We recruit learners of Korean "
      "as a foreign language at CEFR A2–B1 equivalent, block-randomised by placement score "
      "into three arms. Arm A receives a fixed expert-authored item bank in random order. "
      "Arm B receives the same fixed bank with Elo-based adaptive selection, representing "
      "current good practice [18]. Arm C receives the full KO-DRILL system. Arms A and B "
      "share an item bank so that the effect of generation is separable from the effect of "
      "adaptivity. Running this pilot at the full scale implied by Fig. 5 — three arms of "
      "60 — is desirable but not required before the claims in Subsections B through C can "
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
      "larger than the 0.28–0.33 range reported for LLM feedback alone [29]. This is a "
      "known limitation of the pilot, not of the paper's technical claims, which is "
      "exactly why Subsections B through C are designed to stand independently of it."),

("H2", "E. Threats to Validity"),

("P", "Three threats apply regardless of which evaluation is run. Novelty effects may "
      "inflate engagement in any condition using the generative system regardless of "
      "instructional quality, which is why retention at four weeks rather than immediate "
      "engagement is the measure weighted most heavily wherever the pilot is run. "
      "Instructor reviewers cannot be blinded to whether an item was generated, since "
      "generated and authored items differ stylistically, which is why verifier–reviewer "
      "agreement is reported as the primary item-quality measure rather than reviewer "
      "ratings alone. KAGAS was developed from corpora of predominantly non-Indonesian "
      "learners and may transfer imperfectly to Indonesian speakers of Korean specifically, "
      "so error-type distributions should be reported by first-language background rather "
      "than assumed to transfer. Finally, EXAONE's licence restricts commercial "
      "deployment; the Qwen3-generator configuration noted in Section III-C should be "
      "treated as the one to validate if deployment beyond research use is ever intended."),

("H1", "VI. Discussion and Conclusion"),

("P", "The argument of this paper is that the move from fixed item banks to generated ones "
      "is not a straightforward improvement but a change of regime, and that several "
      "methods inherited from the fixed-bank era stop working when it happens. Deep "
      "knowledge tracing loses the dense interaction data it depends on. Classical adaptive "
      "testing loses the pre-calibrated bank it assumes. The Elo rating system, used in the "
      "one configuration a generative pool makes unavoidable, drifts without converging—a "
      "result we have reproduced here at realistic scale, with terminal error growing by a "
      "factor of nearly four from its minimum while rank-order correlation stayed above "
      "0.97, which is to say while the failure stayed invisible to the diagnostics a "
      "practitioner would most naturally check."),

("P", "The remedies we propose are deliberately modest in their use of deep learning. A "
      "small encoder predicts difficulty well enough to initialise a psychometric prior; an "
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
 "S.-H. Hong and S. C. Yang, “Enhancing Korean as a foreign language learning through a mobile-assisted blended course: A triarchic and motivational perspective,” <i>Language Teaching Research</i>, Sep. 2025, doi: 10.1177/13621688251368643.",
 "R. Godwin-Jones, “Distributed agency in second language learning and teaching through generative AI,” <i>Language Learning &amp; Technology</i>, vol. 28, no. 2, pp. 5–31, Jun. 2024, doi: 10.64152/10125/73570.",
 "T.-Y. Tai and H. H.-J. Chen, “Impact of generative AI chatbots and interaction modes on the speaking proficiency of adolescent EFL learners,” <i>Computer Assisted Language Learning</i>, pp. 1–30, Oct. 2025, doi: 10.1080/09588221.2025.2572999.",
 "S. Rashid, S. Malik, and F. Ghauri, “Ethically integrated generative AI for reading and vocabulary development in higher education: An experimental study of efficacy and learner perceptions,” <i>Asian-Pacific Journal of Second and Foreign Language Education</i>, vol. 11, no. 1, Sep. 2026, doi: 10.1186/s40862-026-00451-3.",
 "H. W. Awalurahman, R. F. Aji, and I. Budi, “Transformer and large language models for automatic multiple-choice question generation: A systematic literature review,” <i>IEEE Access</i>, vol. 13, pp. 127100–127112, 2025, doi: 10.1109/ACCESS.2025.3590423.",
 "R. Oliveira, A. Hernández, M. Garbin, J. Dos Reis, and R. Azevedo, “Automatic question generation with large language models: A survey,” <i>ACM Computing Surveys</i>, vol. 58, no. 15, pp. 1–34, Nov. 2026, doi: 10.1145/3832773.",
 "J. Ling and M. Afzaal, “Automatic question-answer pairs generation using pre-trained large language models in higher education,” <i>Computers and Education: Artificial Intelligence</i>, vol. 6, p. 100252, Jun. 2024, doi: 10.1016/j.caeai.2024.100252.",
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
]

FIGURES = [
 ("figures/fig1_architecture.png", 6.3,
  "Fig. 1. The KO-DRILL architecture. Four model roles are kept separate: a Korean generator "
  "(EXAONE), a diagnostic distractor generator, a difficulty-prior encoder (KLUE-RoBERTa), and "
  "an English explainer (Qwen3). Learner state is held by an explicit multidimensional Urnings "
  "tracker rather than by a neural network, so that every ability estimate carries a known "
  "standard error."),
 ("figures/fig2_distractors.png", 6.4,
  "Fig. 2. Misconception-conditioned distractor generation (Novelty 1). Each distractor is "
  "generated from, and tagged with, a specific misconception category drawn from KAGAS, an "
  "independently validated 14-category Korean error taxonomy, so that choosing it updates "
  "only the corresponding skill dimension. The verifier gate guards against the dominant "
  "failure mode of generated distractors—an option that is accidentally also correct—and "
  "a failed item is revised by a critic rather than discarded and resampled blindly."),
 ("figures/fig3_escalation.png", 7.0,
  "Fig. 3. Uncertainty-gated escalation (Novelty 2). Expensive reasoning-mode generation and "
  "contrastive feedback are spent only where the posterior is wide or a misconception has "
  "persisted, rather than uniformly across all interactions."),
 ("figures/fig4_simulation.png", 7.0,
  "Fig. 4. Motivating simulation: 120,000 responses, 300 learners, 400 items, Rasch generating "
  "model, with adaptive selection and simultaneous item calibration — the regime a generative "
  "item pool forces. (a) Elo's rating spread inflates to 2.69 against a true value of 1.04 and "
  "is still rising; Urnings holds at 1.11. (b) Elo's error against true ability grows after "
  "roughly 15,000 responses, so the estimates get worse as more data arrives. This reproduces, "
  "at realistic scale, the failure characterised analytically by Bolsinova et al. [15]."),
 ("figures/fig5_experiment.png", 7.0,
  "Fig. 5. Human study design. Arms A and B share a fixed expert-authored item bank so that the "
  "effect of generation is separable from the effect of adaptivity; Arm C receives the full "
  "system. Retention at four weeks, rather than immediate post-test performance, is the measure "
  "weighted most heavily, to guard against novelty effects."),
]
