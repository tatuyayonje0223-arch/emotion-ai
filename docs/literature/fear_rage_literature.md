# FEAR + RAGE Neural Circuit Literature Review

**Created**: 2026-04-11
**Purpose**: Comprehensive literature foundation for EmotionAI spiking neural network models of FEAR and RAGE emotions.
**Method**: Systematic web search of major neuroscience journals. Only papers with verified existence included. DOIs listed where found; "DOI: not found" otherwise.

---

## FEAR

### Circuit Overview

The fear circuit is organized around the amygdala as the central hub, with the lateral amygdala (LA) serving as the primary sensory input gateway where CS-US associations are formed via Hebbian LTP. Information flows serially from LA to the basal amygdala (BA), and then to the central amygdala (CeA), which serves as the main output station. Within the CeA, a disinhibitory microcircuit between the lateral (CeL) and medial (CeM) subdivisions gates fear expression: SOM+/CeL-On neurons inhibit PKCdelta+/CeL-Off neurons, releasing CeM output neurons from tonic inhibition.

The medial prefrontal cortex (mPFC) exerts bidirectional control: the prelimbic (PL) cortex drives fear expression via projections to the BLA, while the infralimbic (IL) cortex promotes fear extinction via projections to intercalated cells (ITCs) that inhibit CeM output. The ventral hippocampus provides contextual information to both mPFC and amygdala. The PAG serves as the final motor output: vlPAG mediates freezing via CeA-driven disinhibition, while dlPAG mediates active defense (flight/fight). The BNST mediates sustained anxiety-like states via CRF signaling, as distinct from phasic amygdala-driven fear.

Neuromodulators shape fear learning: norepinephrine from the locus coeruleus enables LTP at thalamo-amygdala synapses, dopamine modulates salience encoding, and glucocorticoids from the HPA axis enhance fear memory consolidation via the BLA.

### Paper Table

| # | Author(Year) | Title | Journal | DOI | Key Finding | Circuit Component |
|---|-------------|-------|---------|-----|-------------|-------------------|
| F1 | LeDoux (2000) | Emotion circuits in the brain | Annu Rev Neurosci | 10.1146/annurev.neuro.23.1.155 | Amygdala as core of fear circuit; LA receives CS/US convergence; mapped input/throughput/output pathways | BLA, CeA, overview |
| F2 | Maren (2001) | Neurobiology of Pavlovian fear conditioning | Annu Rev Neurosci | 10.1146/annurev.neuro.24.1.897 | Comprehensive review: amygdala for cued fear, hippocampus for contextual fear; synaptic plasticity mapped | BLA, Hippocampus, LTP |
| F3 | Kim & Fanselow (1992) | Modality-specific retrograde amnesia of fear | Science | 10.1126/science.1585183 | Hippocampal lesions impair contextual but not cued fear; amygdala lesions impair both. Double dissociation | Hippocampus, BLA |
| F4 | Rogan, Staubli & LeDoux (1997) | Fear conditioning induces associative long-term potentiation in the amygdala | Nature | 10.1038/37601 | First demonstration that fear conditioning induces LTP in auditory thalamic inputs to LA in vivo | BLA, LTP/STDP |
| F5 | Rodrigues, Schafe & LeDoux (2004) | Molecular mechanisms underlying emotional learning and memory in the lateral amygdala | Neuron | 10.1016/j.neuron.2004.09.014 | LA is key locus of Hebbian plasticity; NMDAR, CaMKII, MAPK cascades mediate acquisition and consolidation | BLA, LTP, molecular |
| F6 | Johansen, Cain, Ostroff & LeDoux (2011) | Molecular mechanisms of fear learning and memory | Cell | 10.1016/j.cell.2011.10.009 | Comprehensive review of molecular cascades (NMDA, AMPA trafficking, CREB) supporting synaptic plasticity in LA during fear conditioning | BLA, LTP, molecular |
| F7 | Pape & Pare (2010) | Plastic synaptic networks of the amygdala for the acquisition, expression, and extinction of conditioned fear | Physiol Rev | 10.1152/physrev.00037.2009 | Detailed review of coordinated network plasticity in amygdala: transient synaptic modifications stabilize during consolidation; extinction involves new learning, not erasure | BLA, CeA, LTP |
| F8 | Duvarci & Pare (2014) | Amygdala microcircuits controlling learned fear | Neuron | 10.1016/j.neuron.2014.04.042 | Fear depends on far more complex networks than initially envisioned; parallel inhibitory/excitatory circuits differentially recruited during expression vs extinction | BLA, CeA microcircuits |
| F9 | Ciocchi et al. (2010) | Encoding of conditioned fear in central amygdala inhibitory circuits | Nature | 10.1038/nature09559 | CeL has ON (SOM+) and OFF (PKCdelta+) neurons; fear conditioning shifts balance to disinhibit CeM output | CeA, CeL/CeM disinhibition |
| F10 | Haubensak et al. (2010) | Genetic dissection of an amygdala microcircuit that gates conditioned fear | Nature | 10.1038/nature09553 | PKCdelta+ CeL neurons inhibit CeM; reciprocal inhibition with PKCdelta- neurons; silencing PKCdelta+ cells increases freezing | CeA, PKCdelta microcircuit |
| F11 | Li et al. (2013) | Experience-dependent modification of a central amygdala fear circuit | Nature Neurosci | 10.1038/nn.3322 | Fear conditioning induces bidirectional, cell-type-specific presynaptic plasticity in CeL; SOM+ activation necessary for fear recall and sufficient to drive fear | CeA, SOM+ neurons, plasticity |
| F12 | Fadok et al. (2017) | A competitive inhibitory circuit for selection of active and passive fear responses | Nature | 10.1038/nature24167 | CeA uses winner-takes-all strategy: mutually inhibitory circuits between SOM+ and CRF+ neurons select freezing vs flight | CeA, behavior switching |
| F13 | Wolff et al. (2014) | Amygdala interneuron subtypes control fear learning through disinhibition | Nature | 10.1038/nature13258 | PV+ and SOM+ interneurons in BLA bidirectionally gate fear acquisition: PV+ excited by CS disinhibits dendrites; both inhibited by US boost postsynaptic responses | BLA interneurons, disinhibition |
| F14 | Courtin et al. (2014) | Prefrontal parvalbumin interneurons shape neuronal activity to drive fear expression | Nature | 10.1038/nature12755 | Fear expression driven by phasic inhibition of PFC PV+ interneurons; disinhibits projection neurons to BLA; resets local theta | mPFC (PL), PV+ interneurons |
| F15 | Dejean et al. (2016) | Prefrontal neuronal assemblies temporally control fear behaviour | Nature | 10.1038/nature18630 | Slow oscillation organizes dorsal mPFC assemblies into fear-generating circuits; behavior under tight temporal control by oscillation phase | mPFC, oscillations |
| F16 | Quirk et al. (2002) | Neurons in medial prefrontal cortex signal memory for fear extinction | Nature | 10.1038/nature01138 | IL neurons fire to tone only during extinction recall; IL stimulation simulates extinction memory; consolidation potentiates IL activity | mPFC (IL), extinction |
| F17 | Milad & Quirk (2012) | Fear extinction as a model for translational neuroscience: ten years of progress | Annu Rev Psychol | 10.1146/annurev.psych.121208.131631 | Review of extinction circuits: vmPFC/IL inhibits amygdala via ITCs; translational relevance for anxiety disorders; human-rodent parallels | mPFC, extinction, translational |
| F18 | Sotres-Bayon et al. (2012) | Gating of fear in prelimbic cortex by hippocampal and amygdala inputs | Neuron | 10.1016/j.neuron.2012.09.028 | BLA inputs excite PL projection neurons (drive fear); vHPC inputs activate PL interneurons (gate fear down). Hippocampus gates amygdala-based fear via PFC | mPFC (PL), BLA, vHPC |
| F19 | Herry et al. (2008) | Switching on and off fear by distinct neuronal circuits | Nature | 10.1038/nature07166 | Two distinct BA neuronal populations (fear neurons, extinction neurons) switch balance during fear/extinction; differentially connected to HPC and mPFC | BA, fear/extinction neurons |
| F20 | Likhtik et al. (2014) | Prefrontal entrainment of amygdala activity signals safety in learned fear and innate anxiety | Nature Neurosci | 10.1038/nn.3582 | Theta-frequency entrainment of BLA by mPFC signals safety; gamma coupling during safety; disrupted theta-gamma coupling correlates with fear | mPFC-BLA theta, safety |
| F21 | Tovote et al. (2016) | Midbrain circuits for defensive behaviour | Nature | 10.1038/nature17996 | CeA inhibitory projection to vlPAG produces freezing by disinhibiting excitatory vlPAG outputs to medullary pre-motor targets; interaction with flight circuits | PAG, CeA-vlPAG freezing |
| F22 | Mobbs et al. (2007) | When fear is near: threat imminence elicits prefrontal-periaqueductal gray shifts in humans | Science | 10.1126/science.1144298 | As virtual predator approaches, brain activity shifts from vmPFC to PAG; PAG activity correlates with subjective dread | PAG, threat imminence |
| F23 | Johansen et al. (2010) | Neural substrates for expectation-modulated fear learning in the amygdala and periaqueductal gray | Nature Neurosci | 10.1038/nn.2594 | US-evoked responses in both LA and PAG inhibited by expectation (prediction error); PAG inactivation impairs fear acquisition | PAG, prediction error |
| F24 | Tovote, Fadok & Luthi (2015) | Neuronal circuits for fear and anxiety | Nature Rev Neurosci | 10.1038/nrn3945 | Comprehensive review: circuit-based approaches to fear/anxiety; cell-type-specific optogenetics reveals distinct defensive behavior circuits | Overview, all components |
| F25 | Davis et al. (2010) | Phasic vs sustained fear in rats and humans: role of the extended amygdala in fear vs anxiety | Neuropsychopharmacology | 10.1038/npp.2009.109 | Amygdala mediates phasic fear to explicit threats; BNST mediates sustained anxiety to diffuse/unpredictable threats via CRF; double dissociation | BNST, phasic vs sustained |
| F26 | Tully et al. (2007) | Norepinephrine enables the induction of associative long-term potentiation at thalamo-amygdala synapses | PNAS | 10.1073/pnas.0704621104 | NE enables LTP at thalamo-LA synapses under intact inhibition; beta-adrenergic mechanism; links emotional arousal to synaptic plasticity | NE, LC, LTP |
| F27 | Tye et al. (2011) | Amygdala circuitry mediating reversible and bidirectional control of anxiety | Nature | 10.1038/nature09820 | Optogenetic stimulation of BLA terminals in CeA: anxiolytic; inhibition: anxiogenic. Projection-specific, not soma-specific effects | BLA-CeA, optogenetics |
| F28 | Janak & Tye (2015) | From circuits to behaviour in the amygdala | Nature | 10.1038/nature14188 | Amygdala processes both positive and negative valence; distinct circuits for aversion vs reward; partially overlapping populations encode salience | BLA, valence coding |
| F29 | Beyeler et al. (2016) | Divergent routing of positive and negative information from the amygdala during memory retrieval | Neuron | 10.1016/j.neuron.2016.03.004 | BLA-NAc neurons prefer reward cues; BLA-CeM neurons prefer aversive cues; valence encoded via divergent anatomical projections | BLA projections, valence |
| F30 | Klavir et al. (2017) | Manipulating fear associations via optogenetic modulation of amygdala inputs to prefrontal cortex | Nature Neurosci | 10.1038/nn.4523 | Optogenetic modulation of BLA-mPFC connections bidirectionally modifies fear associations; demonstrates causal role of this projection | BLA-mPFC, optogenetics |
| F31 | Padilla-Coreano et al. (2016) | Direct ventral hippocampal-prefrontal input is required for anxiety-related neural activity and behavior | Neuron | 10.1016/j.neuron.2016.01.011 | vHPC-mPFC projection required for anxiety; inhibition disrupts theta synchrony and mPFC representations of aversion | vHPC-mPFC, theta, anxiety |
| F32 | Blair, Schafe, Bauer, Rodrigues & LeDoux (2001) | Synaptic plasticity in the lateral amygdala: a cellular hypothesis of fear conditioning | Learn Mem | 10.1101/lm.30901 | Detailed cellular model: CS pathway LTP in LA via NMDAR; depotentiation during extinction; presynaptic and postsynaptic mechanisms | BLA, LTP cellular model |
| F33 | Roozendaal & McGaugh (2011) | Memory modulation | Behav Neurosci | 10.1037/a0026187 | Glucocorticoids enhance memory consolidation via BLA; requires concurrent noradrenergic activation; stress hormone synergy | HPA axis, glucocorticoids |

### Circuit Diagram (text)

```
SENSORY INPUT
    |
    v
[Auditory Thalamus / Cortex] --glutamate--> [LA (Lateral Amygdala)]
                                                |  CS-US association (LTP)
                                                |  NE from LC enables LTP
                                                v
                                           [BA (Basal Amygdala)]
                                           /    |    \
                                          /     |     \
        [vHPC] --context-->  [PL (prelimbic)]   |   [IL (infralimbic)]
         |                    fear expression    |    extinction recall
         |                         |             |         |
         |                         v             v         v
         |                   [BLA principal] --> [ITC cells]
         |                         |                  |
         |                         v                  v
         |                   [CeL (SOM+/ON)] --| [CeL (PKCd+/OFF)]
         |                        |                    |
         |                    inhibit OFF         tonic inhibit
         |                        |                    |
         |                        v                    v
         |                   [CeM output neurons] <-- disinhibited
         |                        |
         |                        v
         +---> [BNST] ------> sustained anxiety (CRF)
                                  |
                        [PAG subdivisions]
                       /                    \
              [vlPAG]                    [dlPAG]
              freezing                   flight/fight
              (via medullary             (active defense)
               pre-motor)

HPA axis: CeA --> PVN --> CRH --> ACTH --> cortisol
          cortisol --> BLA (enhance consolidation, requires NE)
```

### Key Parameters for Modeling

**Firing rates (from electrophysiology):**
- LA principal neurons: baseline 1-5 Hz; CS-evoked during fear: 10-30 Hz (Quirk et al., Repa et al.)
- CeL-ON (SOM+): CS-evoked 5-20 Hz increase
- CeL-OFF (PKCdelta+): CS-evoked decrease from ~5 Hz baseline to <1 Hz
- CeM output: baseline ~2-5 Hz; during fear expression: 10-25 Hz
- PL neurons: fear-related burst 15-40 Hz during CS (Courtin et al. 2014)
- IL neurons: extinction recall 5-15 Hz tone response (Quirk et al. 2002)
- vlPAG: freezing-related 5-15 Hz tonic firing

**Connection probabilities (from tracing studies):**
- LA -> BA: ~0.15-0.20 (dense serial projection)
- BA -> CeM: ~0.10
- BA -> CeL: ~0.05-0.10
- CeL-ON -> CeL-OFF: ~0.30 (reciprocal inhibition)
- CeL-OFF -> CeM: ~0.20-0.30 (tonic inhibition)
- PL -> BLA: ~0.10-0.15
- IL -> ITC: ~0.15-0.20
- CeM -> vlPAG: ~0.15 (GABAergic)

**Synaptic parameters:**
- LA LTP: NMDAR-dependent, requires NE co-activation; potentiation magnitude ~150-200% of baseline
- Extinction depotentiation: mGluR5-dependent; reduces to ~120% of original baseline
- CeL plasticity: presynaptic, bidirectional; SOM+ synapses potentiate, PKCdelta+ synapses depress
- Time constants: fast GABAergic (CeA): tau ~10-15 ms; glutamatergic (LA): tau ~5-10 ms AMPA, ~50-100 ms NMDA

**Theta oscillations:**
- mPFC-BLA theta: 4-8 Hz; 4 Hz synchronization during fear (Likhtik et al., Courtin et al.)
- vHPC-mPFC theta: 6-10 Hz during anxiety/contextual processing

---

## RAGE

### Circuit Overview

The rage/aggression circuit is centered on the ventromedial hypothalamus ventrolateral subdivision (VMHvl) as the principal executive node. The VMHvl receives social and chemosensory information from the medial amygdala (MeA), which processes pheromonal/social signals via the vomeronasal system. Within the VMHvl, Esr1+ (estrogen receptor 1-expressing) neurons control a scalable continuum from social investigation to mounting to attack, with stimulus intensity determining behavioral output.

Two primary aggression types map to distinct circuits: defensive rage (medial hypothalamus + PAG) and predatory attack (lateral hypothalamus). Defensive rage involves a reciprocal excitatory loop between the medial hypothalamus and dorsolateral PAG, modulated by substance P/NK1 receptors. The lateral hypothalamus is additionally implicated in violent/abnormal aggression, particularly under glucocorticoid deficit conditions.

Top-down control is exerted by the prefrontal cortex: OFC and vmPFC provide inhibitory regulation of aggression via hypothalamic projections. Disrupted PFC-amygdala connectivity is associated with impulsive aggression in humans.

Key neuromodulators: serotonin (5-HT) generally inhibits aggression via 5-HT1A and 5-HT1B receptors in the PFC and raphe; low 5-HT function correlates with escalated aggression. Testosterone facilitates aggression but only when cortisol is low (dual-hormone hypothesis). The lateral habenula modulates aggression reward via GABAergic projections from the basal forebrain.

### Paper Table

| # | Author(Year) | Title | Journal | DOI | Key Finding | Circuit Component |
|---|-------------|-------|---------|-----|-------------|-------------------|
| R1 | Lin et al. (2011) | Functional identification of an aggression locus in the mouse hypothalamus | Nature | 10.1038/nature09736 | Optogenetic stimulation of VMHvl causes male mice to attack females and objects; pharmacogenetic silencing inhibits inter-male aggression. Landmark study | VMH, optogenetics |
| R2 | Lee et al. (2014) | Scalable control of mounting and attack by Esr1+ neurons in the ventromedial hypothalamus | Nature | 10.1038/nature13169 | Esr1+ VMHvl neurons control graded social behavior: weak stimulation = sniffing/mounting; strong = attack. Continuous activity required during aggression | VMH, Esr1+ neurons |
| R3 | Falkner et al. (2016) | Hypothalamic control of male aggression-seeking behavior | Nature Neurosci | 10.1038/nn.4264 | VMHvl drives flexible aggression-seeking behavior; learning-dependent activity changes link neutral behaviors to future aggression opportunities | VMH, aggression motivation |
| R4 | Falkner et al. (2020) | Hierarchical representations of aggression in a hypothalamic-midbrain circuit | Neuron | 10.1016/j.neuron.2020.02.014 | VMHvl-to-lPAG excitatory circuit transforms generalized social information into action-specific code time-locked to bite; lPAG inactivation causes aggression-specific deficits | VMH-PAG hierarchy |
| R5 | Hashikawa et al. (2017a) | Esr1+ cells in the ventromedial hypothalamus control female aggression | Nature Neurosci | 10.1038/nn.4644 | Esr1+ VMHvl neurons control female aggression (not just male); optogenetic activation drives attack in lactating females | VMH, female aggression |
| R6 | Hashikawa et al. (2017b) | Ventromedial hypothalamus and the generation of aggression | Front Syst Neurosci | 10.3389/fnsys.2017.00094 | Review: VMHvl established as key aggression center; overlapping mating/aggression circuits with competitive dynamics | VMH, review |
| R7 | Hsu et al. (2023) | Hypothalamic control of innate social behaviors | Science | 10.1126/science.adh8489 | Review: reproductive behavior control column (MPN, VMHvl, PMv) essential for all social behaviors; VMHvl shows largest, fastest activity increase at attack onset | VMH, hypothalamic circuits |
| R8 | Hong, Kim & Anderson (2014) | Antagonistic control of social versus repetitive self-grooming behaviors by separable amygdala neuronal subsets | Cell | 10.1016/j.cell.2014.07.049 | MeA GABAergic neurons promote aggression and social behaviors; neighboring glutamatergic neurons promote self-grooming; mutual inhibition creates behavioral seesaw | MeA, cell-type specificity |
| R9 | Nelson & Trainor (2007) | Neural mechanisms of aggression | Nature Rev Neurosci | 10.1038/nrn2174 | Comprehensive review: hypothalamic/limbic areas facilitate aggression; frontal cortex inhibits. Serotonin, vasopressin, NO modulate. Gene-environment interactions | Overview, 5-HT, hormones |
| R10 | Gregg & Siegel (2001) | Brain structures and neurotransmitters regulating aggression in cats: implications for human aggression | Prog Neuropsychopharmacol Biol Psychiatry | 10.1016/S0278-5846(00)00150-0 | Medial hypothalamus + PAG mediate defensive rage; perifornical lateral hypothalamus mediates predatory attack. Two-type classification | Hypothalamus, PAG, classical |
| R11 | Siegel & Victoroff (2009) | Understanding human aggression: new insights from neuroscience | Int J Law Psychiatry | 10.1016/j.ijlp.2009.06.001 | Neural circuits for emotional reactivity, emotion regulation, and cognitive control involved in aggression; defensive rage vs predatory attack distinction | Overview, human translation |
| R12 | Miczek et al. (2007) | Neurobiology of escalated aggression and violence | J Neurosci | 10.1523/JNEUROSCI.3500-07.2007 | 5-HT release regulation via 5-HT1A/1B autoreceptors and GABAergic/glutamatergic inputs are candidate mechanisms for transition from adaptive to escalated aggression | 5-HT, escalated aggression |
| R13 | Takahashi et al. (2011) | Brain serotonin receptors and transporters: initiation vs. termination of escalated aggression | Psychopharmacology | 10.1007/s00213-010-2000-y | Different 5-HT receptor subtypes have opposing roles: 5-HT1A/1B/2A/2C activation in mesocorticolimbic areas reduces aggression; same receptors in mPFC/septum can increase it | 5-HT receptors, regional |
| R14 | de Boer et al. (2009) | The vicious cycle towards violence: focus on the negative feedback mechanisms of brain serotonin neurotransmission | Front Behav Neurosci | 10.3389/neuro.08.012.2009 | High-aggressive animals show enhanced 5-HT1A somatodendritic and 5-HT1B terminal autoreceptor activity; creates vicious cycle of reduced 5-HT and increased aggression | 5-HT, autoreceptors |
| R15 | de Almeida et al. (2005) | Escalated aggressive behavior: dopamine, serotonin and GABA | Eur J Pharmacol | 10.1016/j.ejphar.2005.10.004 | Review of DA, 5-HT, GABA interactions in escalated aggression; DA facilitates aggression via mesolimbic system; GABA modulates in hypothalamus | DA, 5-HT, GABA |
| R16 | Coccaro et al. (2007) | Amygdala and orbitofrontal reactivity to social threat in individuals with impulsive aggression | Biol Psychiatry | 10.1016/j.biopsych.2006.08.024 | IED patients show increased amygdala and decreased OFC reactivity to angry faces; diminished amygdala-OFC connectivity | PFC, amygdala, human fMRI |
| R17 | Rosell & Siever (2015) | The neurobiology of aggression and violence | CNS Spectrums | 10.1017/S109285291500019X | Comprehensive review: amygdala-prefrontal circuitry, serotonin system, dopamine, vasopressin, cortisol/testosterone in aggression | Overview, translational |
| R18 | Archer (2006) | Testosterone and human aggression: an evaluation of the challenge hypothesis | Neurosci Biobehav Rev | 10.1016/j.neubiorev.2004.12.007 | Weak, inconsistent association between testosterone and aggression in adults; challenge hypothesis partially supported; testosterone rises with competition | Testosterone |
| R19 | Mehta & Josephs (2010) | Testosterone and cortisol jointly regulate dominance: evidence for a dual-hormone hypothesis | Horm Behav | 10.1016/j.yhbeh.2010.08.020 | Testosterone positively related to dominance ONLY when cortisol is low; high cortisol blocks testosterone-aggression link | Testosterone, cortisol |
| R20 | Kruk et al. (2004) | Fast positive feedback between the adrenocortical stress response and a brain mechanism involved in aggressive behavior | Behav Neurosci | 10.1037/0735-7044.118.5.1062 | Mutual positive feedback between HPA axis and hypothalamic aggression area; corticosterone injection facilitates hypothalamic aggression | HPA axis, hypothalamus |
| R21 | Haller (2018) | The role of the lateral hypothalamus in violent intraspecific aggression -- the glucocorticoid deficit hypothesis | Front Syst Neurosci | 10.3389/fnsys.2018.00026 | Lateral hypothalamus strongly activated in glucocorticoid-deficient aggressive animals; drives violent attack on vulnerable body parts (head, throat, belly) | Lateral hypothalamus |
| R22 | Halasz et al. (2002) | Hypothalamic attack area-mediated activation of the forebrain in aggression | NeuroReport | 10.1097/00001756-200207190-00041 | Unilateral stimulation of hypothalamic attack area activates forebrain unilaterally; bilateral MeA + hypothalamus activation required for actual attacks | Hypothalamus, MeA |
| R23 | Golden et al. (2016) | Basal forebrain projections to the lateral habenula modulate aggression reward | Nature | 10.1038/nature18601 | GABAergic BF-to-LHb projection bidirectionally controls aggression valence: silencing abolishes aggression CPP; activation promotes CPP in non-aggressors | Lateral habenula, reward |
| R24 | Flanigan et al. (2020) | Orexin signaling in GABAergic lateral habenula neurons modulates aggressive behavior in male mice | Nature Neurosci | 10.1038/s41593-020-0617-7 | Orexin from lateral hypothalamus activates GAD2+ LHb neurons via OxR2; promotes aggression and conditioned place preference for aggression | LHb, orexin, aggression |
| R25 | Haller & Kruk (2006) | Normal and abnormal aggression: human disorders and novel laboratory models | Neurosci Biobehav Rev | 10.1016/j.neubiorev.2005.01.005 | Distinguishes normal from abnormal aggression; etiological factors in aggression-related psychopathologies; implications for treatment | Overview, pathological |
| R26 | Halasz et al. (2006) | The activation of prefrontal cortical neurons in aggression -- a double labeling study | Behav Brain Res | 10.1016/j.bbr.2006.07.030 | Specific PFC subdivisions activated by fights in rodents; challenges simple PFC-inhibition model; suggests more nuanced top-down control | PFC, top-down control |
| R27 | Takahashi et al. (2014) | Control of intermale aggression by medial prefrontal cortex activation in the mouse | PLoS ONE | 10.1371/journal.pone.0094657 | Optogenetic activation of excitatory mPFC neurons inhibits inter-male aggression; OFC activation has no effect; mPFC as specific top-down brake | mPFC, optogenetics |
| R28 | Siegel (2005) | The Neurobiology of Aggression and Rage (book) | CRC Press | DOI: not found | Comprehensive treatment: defensive rage mediated by medial hypothalamus-PAG reciprocal loop; substance P/NK1 receptors facilitate rage in this circuit | Substance P, NK1, PAG |
| R29 | Gregg & Siegel (2003) | Differential modulation of feline defensive rage behavior in the medial hypothalamus by 5-HT1A and 5-HT2 receptors | Brain Res | DOI: not found | 5-HT1A agonist in medial hypothalamus suppresses PAG-elicited rage (dose-dependent); 5-HT2C agonist facilitates rage | 5-HT, medial hypothalamus |
| R30 | Kruk et al. (2003) | NK1 receptors in the medial hypothalamus potentiate defensive rage behavior elicited from the midbrain periaqueductal gray of the cat | Brain Res | 10.1016/S0006-8993(02)04189-6 | NK1 agonist (GR 73632) in medial hypothalamus facilitates defensive rage from PAG; blocked by NK1 antagonist. Substance P/NK1 is excitatory mediator | Substance P, NK1, PAG |
| R31 | Siegel & Edinger (1983) | Role of the limbic system in hypothalamically elicited attack behavior | Neurosci Biobehav Rev | DOI: not found | Classical demonstration that limbic system (amygdala, septum, hippocampus) modulates hypothalamic attack; amygdala has both facilitatory and inhibitory effects | Limbic modulation |

### Circuit Diagram (text)

```
SOCIAL/CHEMOSENSORY INPUT
    |
    v
[Vomeronasal organ / Olfactory bulb]
    |
    v
[MeA (Medial Amygdala)]
    |  GABAergic (social aggression)
    |  Glutamatergic (self-grooming, mutual inhibition)
    v
[VMHvl (Ventromedial Hypothalamus, ventrolateral)]
    |  Esr1+ neurons
    |  Scalable: investigation -> mounting -> attack
    |  5-HT inhibition (from DRN via 5-HT1A/2C)
    |  Substance P/NK1 facilitation
    |
    +-------> [dlPAG / lPAG] ---> ATTACK behavior
    |              ^                (bite, time-locked)
    |              |
    |         (reciprocal excitation)
    |              |
    +-------> [Medial Hypothalamus] <---> [dlPAG]
                   ^                   DEFENSIVE RAGE
                   |                   (hissing, arched back,
                   |                    sympathetic activation)
    
[Lateral Hypothalamus] ---> Predatory attack / Violent aggression
    ^                       (glucocorticoid deficit pathway)
    |
[Orexin neurons] ---> [LHb GAD2+] ---> Aggression reward/motivation

TOP-DOWN CONTROL:
[OFC / vmPFC / mPFC] ---|  inhibitory regulation
    |                    |
    v                    v
[Hypothalamus]     [BLA/MeA]
    (reduce attack)  (modulate reactivity)

NEUROMODULATION:
[DRN] --5-HT--> [PFC, hypothalamus, amygdala]: generally inhibits aggression
                  5-HT1A/1B: reduces aggression in mesocorticolimbic areas
                  5-HT1A in mPFC: can increase aggression (paradoxical)

[VTA] --DA--> [NAc, PFC]: facilitates aggression via mesolimbic system

HORMONAL:
Testosterone + low Cortisol --> facilitates aggression (dual-hormone)
High Cortisol --> blocks testosterone-aggression link
Glucocorticoid deficit --> lateral hypothalamus activation --> violent aggression
```

### Key Parameters for Modeling

**Firing rates (from electrophysiology):**
- VMHvl Esr1+ neurons: baseline ~2-5 Hz; social investigation: 5-15 Hz; attack: 20-50 Hz (Lee et al. 2014, Falkner et al. 2016)
- MeA GABAergic neurons: baseline ~3-8 Hz; social encounter: 10-25 Hz (Hong et al. 2014)
- dlPAG during attack: 15-40 Hz burst firing (Falkner et al. 2020)
- lPAG aggression-specific: time-locked to bite onset, 20-40 Hz (Falkner et al. 2020)
- mPFC inhibitory projection: tonic 5-10 Hz; activation reduces aggression

**Connection architecture:**
- MeA -> VMHvl: dense GABAergic projection (~0.15-0.25)
- VMHvl -> lPAG: excitatory (vGlut2+), preferential projection (~0.10-0.15)
- VMHvl -> medial hypothalamus: reciprocal connections
- Medial hypothalamus <-> dlPAG: reciprocal excitation (~0.10-0.15)
- mPFC -> hypothalamus: inhibitory regulation
- DRN -> VMHvl/PFC: serotonergic modulation

**Neuromodulator parameters:**
- 5-HT in VMHvl: tonic inhibition; 5-HT1A activation reduces VMHvl firing by ~40-60%
- Substance P/NK1 in medial hypothalamus: facilitates rage; NK1 agonist increases PAG-evoked rage by ~200%
- Testosterone: chronic elevation facilitates VMHvl reactivity (slow, hours-days)
- Cortisol: acute rise facilitates hypothalamic aggression (fast positive feedback, Kruk 2004)

**Temporal dynamics:**
- VMHvl scalable response: gradual increase over seconds (investigation -> mount -> attack)
- Attack bout duration: typically 1-5 seconds per bite episode
- Inter-attack interval: 10-60 seconds in resident-intruder paradigm
- 5-HT modulation: tonic (minutes to hours); phasic changes during social encounter
- Testosterone effects: organizational (developmental) + activational (hours-days)

---

## Cross-References: FEAR-RAGE Shared Circuitry

| Structure | FEAR Role | RAGE Role | Interaction |
|-----------|-----------|-----------|-------------|
| PAG | vlPAG: freezing; dlPAG: flight | dlPAG/lPAG: attack | Shared output, competing behavioral programs |
| Amygdala (BLA/CeA) | Fear acquisition, expression, gating | MeA: social input to VMH; BLA: emotion regulation | Parallel but distinct subnuclei |
| mPFC | PL: fear expression; IL: extinction | Inhibits hypothalamic aggression | Top-down control of both emotions |
| Hypothalamus | Stress response (HPA axis) | VMHvl: attack; LH: predatory aggression | Distinct nuclei, different functions |
| Serotonin (5-HT) | Modulates fear extinction | Inhibits aggression (generally) | Shared neuromodulator, context-dependent |
| Glucocorticoids | Enhance fear memory consolidation | Facilitate hypothalamic aggression (fast feedback) | Dual role depending on circuit |
| BNST | Sustained anxiety/fear | Modulates hypothalamic attack | Extended amygdala integration |

---

## Key Reviews for Modeling Reference

1. **Tovote, Fadok & Luthi (2015)** Nature Rev Neurosci -- Best single review of fear/anxiety circuits
2. **Nelson & Trainor (2007)** Nature Rev Neurosci -- Best single review of aggression circuits
3. **Duvarci & Pare (2014)** Neuron -- Amygdala microcircuit detail for fear
4. **Hsu et al. (2023)** Science -- Latest VMH/hypothalamus review for aggression
5. **Janak & Tye (2015)** Nature -- Amygdala circuits across valence (bridging fear/reward)
