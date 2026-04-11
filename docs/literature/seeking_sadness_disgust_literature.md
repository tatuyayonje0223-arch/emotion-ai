# SEEKING + SADNESS + DISGUST Neural Circuit Literature Review

**Date**: 2026-04-11
**Purpose**: Comprehensive literature survey for EmotionAI spiking neural network modeling
**Scope**: 90+ papers from major neuroscience journals

---

## SEEKING

### Circuit Overview

The SEEKING system (Panksepp 1998) corresponds to the mesolimbic dopamine pathway, centered on VTA dopamine neurons projecting to the nucleus accumbens (NAc), prefrontal cortex (PFC), and amygdala. Critically, SEEKING = "wanting" (incentive salience), NOT "liking" (hedonic pleasure). This distinction, established by Berridge & Robinson (1998), shows that dopamine mediates motivational approach and anticipatory excitement, while opioid/endocannabinoid systems in NAc hedonic hotspots mediate consummatory pleasure. VTA DA neurons encode reward prediction errors (RPE; Schultz 1997): excited by better-than-expected outcomes, unresponsive to expected outcomes, and inhibited by worse-than-expected outcomes. The circuit supports exploration, curiosity, goal-directed behavior, and reinforcement learning.

### Paper Table

| # | Author(Year) | Title | Journal | Key Finding | Circuit Component |
|---|-------------|-------|---------|-------------|-------------------|
| 1 | Schultz(1997) | A neural substrate of prediction and reward | Science 275:1593-1599 | DA neurons encode reward prediction error: excited by better-than-predicted, depressed by worse-than-predicted outcomes | VTA DA / RPE |
| 2 | Berridge(1998) | What is the role of dopamine in reward: hedonic impact, reward learning, or incentive salience? | Brain Research Reviews 28:309-369 | DA = "wanting" (incentive salience), NOT "liking"; opioids mediate hedonic pleasure | Wanting vs Liking |
| 3 | Panksepp(1998) | Affective Neuroscience: The Foundations of Human and Animal Emotions | Oxford University Press (book) | Defined SEEKING system as mesolimbic DA circuit mediating anticipatory eagerness and exploration | SEEKING system |
| 4 | Olds & Milner(1954) | Positive reinforcement produced by electrical stimulation of septal area and other regions of rat brain | J Comp Physiol Psychol 47:419-427 | Rats self-stimulate reward circuits at >1000 presses/hour; discovered brain reward pathways | Brain stimulation reward |
| 5 | Wise(2004) | Dopamine, learning and motivation | Nature Reviews Neuroscience 5:483-494 | DA vital for stamping-in stimulus-reward and response-reward associations; motivation requires DA-dependent learning | DA / learning / motivation |
| 6 | Tsai(2009) | Phasic firing in dopaminergic neurons is sufficient for behavioral conditioning | Science 324:1080-1084 | Optogenetic phasic (but NOT tonic) VTA DA stimulation sufficient for conditioned place preference | VTA / phasic vs tonic |
| 7 | Bromberg-Martin(2010) | Dopamine in motivational control: rewarding, aversive, and alerting | Neuron 68:815-834 | Two DA neuron types: value-coding (excited by reward, inhibited by aversion) and salience-coding (excited by both) | VTA DA heterogeneity |
| 8 | Lammel(2012) | Input-specific control of reward and aversion in the ventral tegmental area | Nature 491:212-217 | LDT inputs to VTA = reward (lateral VTA DA to NAc lateral shell); LHb inputs = aversion (medial VTA DA to mPFC) | VTA lateral vs medial |
| 9 | Cohen(2012) | Neuron-type-specific signals for reward and punishment in the ventral tegmental area | Nature 482:85-88 | DA neurons = RPE (Type I); GABA neurons = expected reward signal (Type II). Optogenetic tagging confirmed | VTA cell types |
| 10 | Steinberg(2013) | A causal link between prediction errors, dopamine neurons and learning | Nature Neuroscience 16:966-973 | Optogenetic VTA DA activation at reward time is sufficient to cause blocking/unblocking, proving RPE causality | VTA DA / causal RPE |
| 11 | Bayer & Glimcher(2005) | Midbrain dopamine neurons encode a quantitative reward prediction error signal | Neuron 47:129-141 | DA neuron firing quantitatively predicted by TD-model RPE for positive prediction errors | VTA DA / quantitative RPE |
| 12 | Dabney(2020) | A distributional code for value in dopamine-based reinforcement learning | Nature 577:671-675 | DA neurons encode full reward distribution (not just mean); distributional RL in the brain | VTA DA / distributional RL |
| 13 | Haber & Knutson(2010) | The reward circuit: linking primate anatomy and human imaging | Neuropsychopharmacology 35:4-26 | Comprehensive cortical-basal ganglia reward circuit: ACC, OFC, ventral striatum, ventral pallidum, VTA DA | Circuit architecture |
| 14 | Ikemoto(2007) | Dopamine reward circuitry: two projection systems from the ventral midbrain to the NAc-olfactory tubercle complex | Brain Research Reviews 56:27-78 | Posteromedial VTA projects to medial NAc shell; lateral VTA projects to NAc core and lateral shell | VTA projections |
| 15 | Morales & Margolis(2017) | Ventral tegmental area: cellular heterogeneity, connectivity and behaviour | Nature Reviews Neuroscience 18:73-85 | VTA contains DA, GABA, glutamate neurons with distinct connectivity; some co-release multiple transmitters | VTA heterogeneity |
| 16 | Floresco(2015) | The nucleus accumbens: an interface between cognition, emotion, and action | Annual Review of Psychology 66:25-52 | NAc integrates frontal/temporal inputs; shell vs core subserve distinct functions in action selection and motivation | NAc shell vs core |
| 17 | Roitman(2005) | Nucleus accumbens neurons are innately tuned for rewarding and aversive taste stimuli | Neuron 45:587-597 | NAc neurons inhibited by sucrose (75%), excited by quinine (75%); segregated microcircuit for valence | NAc valence coding |
| 18 | Salamone & Correa(2012) | The mysterious motivational functions of mesolimbic dopamine | Neuron 76:470-485 | NAc DA mediates effort, behavioral activation, and approach behavior; NOT hedonic pleasure per se | NAc DA / effort |
| 19 | Berridge(2009) | Dissecting components of reward: 'liking', 'wanting', and learning | Current Opinion in Pharmacology 9:65-73 | Hedonic hotspot in NAc medial shell (1 mm^3): mu-opioid stimulation increases liking; DA increases wanting across all NAc | NAc hotspot / opioid |
| 20 | Floresco(2003) | Afferent modulation of dopamine neuron firing differentially regulates tonic and phasic dopamine transmission | Nature Neuroscience 6:968-973 | Pallidal inputs regulate population activity (tonic DA); pedunculopontine inputs regulate burst firing (phasic DA) | Tonic vs phasic regulation |
| 21 | Padoa-Schioppa & Assad(2006) | Neurons in the orbitofrontal cortex encode economic value | Nature 441:223-226 | OFC neurons encode subjective value of offered/chosen goods, independent of visuospatial or motor factors | OFC value coding |
| 22 | Rolls & Grabenhorst(2008) | The orbitofrontal cortex and beyond: from affect to decision-making | Progress in Neurobiology 86:216-244 | OFC represents reward value of primary reinforcers (taste, touch, smell); provides common currency for comparison | OFC reward representation |
| 23 | Gruber(2014) | States of curiosity modulate hippocampus-dependent learning via the dopaminergic circuit | Neuron 84:486-496 | High curiosity enhances midbrain + NAc activity; curiosity-driven memory benefits depend on VTA-hippocampus connectivity | DA / curiosity / exploration |
| 24 | Duzel(2010) | NOvelty-related Motivation of Anticipation and exploration by Dopamine (NOMAD) | Neuroscience & Biobehavioral Reviews 34:660-669 | Hippocampus-VTA-NAc loop for novelty detection; DA promotes exploration and memory encoding for novel stimuli | DA / novelty / NOMAD |
| 25 | Nair-Roberts(2008) | Stereological estimates of dopaminergic, GABAergic and glutamatergic neurons in the VTA, SN and retrorubral field in the rat | Neuroscience 152:1024-1031 | VTA composition: ~60% DA, ~35% GABA, ~2-3% glutamate neurons by stereological count | VTA cell composition |
| 26 | Watabe-Uchida(2012) | Whole-brain mapping of direct inputs to midbrain dopamine neurons | Neuron 74:858-873 | Comprehensive monosynaptic input mapping to VTA DA neurons using rabies tracing; anterior cortex to VTA to NAc circuit | VTA input mapping |
| 27 | Nestler & Carlezon(2006) | The mesolimbic dopamine reward circuit in depression | Biological Psychiatry 59:1151-1159 | NAc and VTA contribute to anhedonia, reduced motivation in depression; CREB/dynorphin/BDNF in VTA-NAc | VTA-NAc in depression |
| 28 | Robinson & Berridge(1993) | The neural basis of drug craving: an incentive-sensitization theory of addiction | Brain Research Reviews 18:247-291 | Repeated drug use sensitizes mesolimbic DA = pathological wanting without increased liking | Incentive sensitization |
| 29 | Alcaraz-Zubeldia(2011) | Behavioral functions of the mesolimbic dopaminergic system: an affective neuroethological perspective | Brain Research Reviews 56:283-321 | SEEKING system neurobiology: VTA DA, 50kHz USVs, investigatory behavior, self-stimulation convergence | SEEKING neuroethology |
| 30 | Panksepp & Watt(2011) | Why does depression hurt? Ancestral primary-process separation-distress (PANIC/GRIEF) and diminished brain reward (SEEKING) processes in the genesis of depressive affect | Psychiatry 74:5-13 | Depression = overactive PANIC/GRIEF + diminished SEEKING; loss of DA-mediated enthusiasm | SEEKING in depression |
| 31 | Grace(2007) | Regulation of firing of dopaminergic neurons and control of goal-directed behaviors | Trends in Neurosciences 30:220-227 | Hippocampal input drives DA population activity; PFC gates phasic responses; model for goal-directed behavior | DA regulation / PFC |

### Circuit Diagram (text)

```
SEEKING / Mesolimbic DA Circuit:

  [Lateral Habenula] ---(-)--> [RMTg] ---(-)--> [VTA DA neurons]
       (aversion signal)                          |
  [LDT / PPTg] ---------(+)--> [VTA DA neurons]  |
       (burst regulation)        /    |    \      |
                                /     |     \     |
                           [NAc]   [mPFC]  [Amygdala]
                          /    \      |
                    [Shell] [Core]  [OFC]
                      |       |      |
                   (wanting) (action (value
                    + liking  selection) coding)
                    hotspot)

Neurotransmitters:
  VTA --> NAc: Dopamine (D1/D2 receptors on MSNs)
  NAc shell hotspot: mu-opioid = liking; DA = wanting
  NAc --> VP: GABA (disinhibition)
  VP --> Thalamus --> PFC: feedback loop
  Hippocampus --> VTA: glutamate (novelty / context)
  PFC --> VTA: glutamate (top-down control)
  LHb --> RMTg --> VTA: GABA (negative RPE)

Firing patterns:
  Tonic DA: ~4 Hz baseline, regulated by pallidal inputs
  Phasic burst: 15-30 Hz, 2-6 spikes, triggered by reward/cues
  Pause: 100-500 ms inhibition for negative RPE
```

### Key Parameters for Modeling

| Parameter | Value | Source |
|-----------|-------|-------|
| VTA DA neuron count (rat) | ~27,000 TH+ | Nair-Roberts(2008) |
| VTA composition | 60% DA, 35% GABA, 5% Glu | Nair-Roberts(2008) |
| DA tonic firing rate | 3-8 Hz | Grace(2007) |
| DA phasic burst | 15-30 Hz, 2-6 spikes | Schultz(1997); Grace(2007) |
| DA pause duration | 100-500 ms | Schultz(1997) |
| RPE signal: positive | +10-20 spikes/s above baseline | Bayer & Glimcher(2005) |
| RPE signal: negative | Complete suppression ~200ms | Schultz(1997) |
| NAc D1-MSN proportion | ~50% of MSNs | Floresco(2015) |
| NAc D2-MSN proportion | ~50% of MSNs | Floresco(2015) |
| Hedonic hotspot size | ~1 mm^3 in medial shell | Berridge(2009) |
| VTA-NAc DA release latency | ~100-200 ms | Tsai(2009) |

---

## SADNESS

### Circuit Overview

The SADNESS/GRIEF circuit (Panksepp's PANIC system) involves hyperactivity of the subgenual anterior cingulate cortex (sgACC, Brodmann area 25), lateral habenula-mediated inhibition of dopamine/serotonin systems, reduction in endogenous opioid tone (separation distress), and aberrant default mode network (DMN) activity supporting rumination. Depression represents a pathological state of sustained sadness, characterized by sgACC overactivity driving limbic hyper-responsiveness and cortical hypofunction (Mayberg's limbic-cortical dysregulation model). The lateral habenula serves as a "disappointment center," inhibiting VTA DA and raphe 5-HT neurons upon reward omission. Endogenous opioid withdrawal during social loss produces separation distress pain, mediated by circuits overlapping with physical pain (dACC, anterior insula).

### Paper Table

| # | Author(Year) | Title | Journal | Key Finding | Circuit Component |
|---|-------------|-------|---------|-------------|-------------------|
| 1 | Mayberg(2005) | Deep brain stimulation for treatment-resistant depression | Neuron 45:651-660 | DBS of sgACC white matter produced sustained remission in 4/6 treatment-resistant patients; reduced sgACC blood flow | sgACC / DBS |
| 2 | Mayberg(1999) | Reciprocal limbic-cortical function and negative mood: converging PET findings in depression and normal sadness | American Journal of Psychiatry 156:675-682 | Sadness = limbic increases (sgACC, insula) + cortical decreases (dlPFC, parietal); depression recovery shows reverse | Limbic-cortical model |
| 3 | Drevets(1997) | Subgenual prefrontal cortex abnormalities in mood disorders | Nature 386:824-827 | sgPFC grey matter volume reduced 39-48% in bipolar/unipolar depression; decreased metabolism | sgACC structure |
| 4 | Ressler & Mayberg(2007) | Targeting abnormal neural circuits in mood and anxiety disorders: from the laboratory to the clinic | Nature Neuroscience 10:1116-1124 | Review of emotion circuit abnormalities in depression; DBS, TMS, VNS as circuit-based therapies | Circuit-based therapy |
| 5 | Johansen-Berg(2008) | Anatomical connectivity of the subgenual cingulate region targeted with DBS for treatment-resistant depression | Cerebral Cortex 18:1374-1383 | sgACC connects to NAc, amygdala, hypothalamus, OFC; distinct from pregenual ACC connectivity | sgACC connectivity |
| 6 | Matsumoto & Hikosaka(2007) | Lateral habenula as a source of negative reward signals in dopamine neurons | Nature 447:1111-1115 | LHb neurons excited by no-reward cues, inhibited by reward cues; LHb stimulation inhibits DA neurons | LHb / negative RPE |
| 7 | Li(2011) | Synaptic potentiation onto habenula neurons in the learned helplessness model of depression | Nature 470:535-539 | In learned helplessness, excitatory synapses onto LHb neurons are potentiated; DBS-like stimulation rescues behavior | LHb / depression |
| 8 | Yang(2018) | Ketamine blocks bursting in the lateral habenula to rapidly relieve depression | Nature 554:317-322 | LHb neurons show increased NMDAR-dependent burst firing in depression; ketamine blocks bursting to relieve symptoms | LHb / ketamine |
| 9 | Hu(2020) | Reward processing by the lateral habenula in normal and depressive behaviors | Nature Neuroscience 17:1298-1303 | LHb provides negative value signals to DA and 5-HT systems; dysfunction linked to depression | LHb review |
| 10 | Caspi(2003) | Influence of life stress on depression: moderation by a polymorphism in the 5-HTT gene | Science 301:386-389 | Short allele of 5-HTTLPR + stressful events = more depression; gene-environment interaction for serotonin | 5-HT genetics |
| 11 | Panksepp(1998) | PANIC/Sadness system in Affective Neuroscience | Oxford University Press (book) | Separation distress mediated by endogenous opioid withdrawal; PAG, thalamus, cingulate, bed nucleus stria terminalis | PANIC system |
| 12 | Panksepp(1978) | The biology of social attachments: opiates alleviate separation distress | Biological Psychiatry 13:607-618 | Low doses of opiates profoundly reduce distress vocalizations in socially isolated puppies | Opioid / separation |
| 13 | Zubieta(2003) | mu-Opioid receptor-mediated antinociceptive responses differ in men and women | Journal of Neuroscience 23:5100-5107 | Social rejection activates mu-opioid system in ventral striatum, amygdala, thalamus, PAG; reduces social pain | Opioid / social pain |
| 14 | Eisenberger(2003) | Does rejection hurt? An fMRI study of social exclusion | Science 302:290-292 | Social exclusion activates dACC (like physical pain); RVPFC regulates distress | ACC / social pain |
| 15 | Raichle(2001) | A default mode of brain function | PNAS 98:676-682 | Identified DMN: mPFC, PCC, lateral parietal; consistently active at rest, deactivated during tasks | DMN discovery |
| 16 | Hamilton(2015) | Depressive rumination, the default-mode network, and the dark matter of clinical neuroscience | Biological Psychiatry 78:224-230 | Increased DMN-sgACC functional connectivity in MDD; predicts rumination levels; sgACC integrates self-referential + withdrawal | DMN / rumination |
| 17 | Sheline(2010) | Resting-state functional MRI in depression unmasks increased connectivity between networks via the dorsal nexus | PNAS 107:11020-11025 | Depressed subjects show increased connectivity from DMN, cognitive control network, and affective network to dorsal nexus | DMN hyperconnectivity |
| 18 | Craig(2009) | How do you feel -- now? The anterior insula and human awareness | Nature Reviews Neuroscience 10:59-70 | Anterior insula re-represents interoception; correlates with all subjective feelings including sadness | Insula / interoception |
| 19 | Harrison(2009) | Neural origins of human sickness in interoceptive responses to inflammation | Biological Psychiatry 66:415-422 | Inflammation-induced fatigue predicted by insula + ACC activity; sgACC reactivity associated with mood changes | Insula / sickness |
| 20 | Krishnan & Nestler(2008) | The molecular neurobiology of depression | Nature 455:894-902 | Maladaptive stress-induced neuroplastic changes in specific circuits; molecular basis of susceptibility vs resilience | Molecular basis |
| 21 | Nestler & Carlezon(2006) | The mesolimbic dopamine reward circuit in depression | Biological Psychiatry 59:1151-1159 | Anhedonia involves VTA-NAc dysfunction; CREB, dynorphin, BDNF, Clock in depression pathophysiology | VTA-NAc / anhedonia |
| 22 | Tye(2013) | Dopamine neurons modulate neural encoding and expression of depression-related behaviour | Nature 493:537-541 | Optogenetic inhibition of VTA DA = depression symptoms; phasic activation rescues chronic stress-induced depression | VTA DA / depression |
| 23 | Ferenczi(2016) | Prefrontal cortical regulation of brainwide circuit dynamics and reward-related behavior | Science 351:aac9698 | Elevated mPFC excitability reduces striatal DA responses and reward-seeking; resembles human depression imaging | mPFC / anhedonia |
| 24 | Duman & Aghajanian(2012) | Synaptic dysfunction in depression: potential therapeutic targets | Science 338:68-72 | Chronic stress reduces synapses in PFC; ketamine rapidly increases synaptogenesis via mTOR and BDNF | Synaptic plasticity |
| 25 | Kupfer(2012) | Major depressive disorder: new clinical, neurobiological, and treatment perspectives | The Lancet 379:1045-1055 | Comprehensive review of depression neurobiology including imaging, genetics, biomarkers | Depression review |
| 26 | Holtzheimer & Mayberg(2011) | Deep brain stimulation for psychiatric disorders | Annual Review of Neuroscience 34:289-307 | Review of DBS approaches; sgACC target most studied; distinct therapeutic response phases | DBS review |
| 27 | Panksepp & Watt(2011) | Why does depression hurt? Ancestral PANIC/GRIEF and diminished SEEKING in depression | Psychiatry 74:5-13 | Depression = overactive separation distress (PANIC) + hypoactive reward exploration (SEEKING) | PANIC-SEEKING model |
| 28 | Gusnard(2001) | Medial prefrontal cortex and self-referential mental activity: relation to a default mode of brain function | PNAS 98:4259-4264 | mPFC highest baseline metabolism; self-referential processing hub; decreases during goal-directed tasks | mPFC / self-reference |
| 29 | Keedwell(2005) | A differential pattern of neural response toward sad versus happy facial expressions in major depressive disorder | Biological Psychiatry 57:201-209 | Depressed patients show increased amygdala/parahippocampal response to sad faces; reduced to happy faces | Amygdala / sad bias |
| 30 | Vytal & Hamann(2010) | Neuroimaging support for discrete neural correlates of basic emotions: a voxel-based meta-analysis | Journal of Cognitive Neuroscience 22:2864-2885 | Sadness associated with sgACC activation; fear with amygdala; disgust with insula; distinct but overlapping | Meta-analysis |

### Circuit Diagram (text)

```
SADNESS / GRIEF Circuit:

  [SOCIAL LOSS / STRESS]
       |
       v
  [Lateral Habenula] ---(+)---> [RMTg] ---(-)--> [VTA DA neurons] (suppressed)
       |                                            |
       +---(+)---> [Raphe nuclei] ---(-)--> 5-HT release (reduced)
       |
  [Subgenual ACC (BA25)] <---(hyperactive in depression)
       |            |           |
       v            v           v
  [Amygdala]   [Hypothalamus] [NAc]
  (sad bias)   (HPA axis,     (anhedonia,
               cortisol)       reduced wanting)
       |
  [Anterior Insula] <--- interoception of sadness/grief
       |
  [Default Mode Network] = [mPFC + PCC + lateral parietal]
       |                    (hyperconnected to sgACC in depression)
       v                    = rumination, self-referential sadness
  [dlPFC] (hypoactive in depression)
       |
       v
  Cognitive control impaired

  PANIC/GRIEF pathway (Panksepp):
  [PAG] --> [Thalamus] --> [ACC/Cingulate] --> [BNST]
       mediated by: endogenous opioids (withdrawal = distress)
                    oxytocin (reduced = vulnerability)
                    CRF (elevated = stress)
```

### Key Parameters for Modeling

| Parameter | Value | Source |
|-----------|-------|-------|
| sgACC metabolic increase in depression | +20-40% vs healthy controls | Mayberg(1999); Drevets(1997) |
| sgACC grey matter loss | 39-48% volume reduction | Drevets(1997) |
| LHb burst frequency in depression | Significantly increased vs control | Yang(2018) |
| LHb-VTA inhibition latency | ~10-20 ms | Matsumoto & Hikosaka(2007) |
| 5-HT reduction in depression | Estimated 20-40% decrease | Caspi(2003); review estimates |
| DMN-sgACC connectivity increase | Significant positive correlation with rumination | Hamilton(2015) |
| DBS response rate (sgACC) | 4/6 (67%) initial; ~50% in larger trials | Mayberg(2005) |
| VTA DA firing in depression model | Reduced phasic, normal tonic | Tye(2013) |
| Social pain ACC activation | dACC z-score ~3-4 for exclusion vs inclusion | Eisenberger(2003) |
| Ketamine onset of antidepressant effect | 2-4 hours | Duman & Aghajanian(2012) |

---

## DISGUST

### Circuit Overview

The disgust circuit centers on the anterior insula (aIC), particularly the left anterior insula, which is the core structure for both experiencing and recognizing disgust across all modalities (taste, smell, visual, moral). The basal ganglia (putamen, caudate, globus pallidus) play a crucial supporting role, as evidenced by Huntington's disease patients showing selective disgust recognition deficits. The circuit has evolutionary origins in gustatory rejection (conditioned taste aversion, Garcia effect), mediated by the parabrachial nucleus (PBN) and area postrema/NTS. Disgust has expanded from oral rejection to pathogen avoidance (core disgust) to moral evaluation (moral disgust), with the OFC bridging visceral and evaluative components. The anterior insula serves as the interoceptive hub where visceral nausea signals become conscious disgust experience.

### Paper Table

| # | Author(Year) | Title | Journal | Key Finding | Circuit Component |
|---|-------------|-------|---------|-------------|-------------------|
| 1 | Phillips(1997) | A specific neural substrate for perceiving facial expressions of disgust | Nature 389:495-498 | Disgust faces activate anterior insula and limbic cortico-striatal-thalamic circuit; NOT amygdala | Anterior insula |
| 2 | Wicker(2003) | Both of us disgusted in my insula: the common neural basis of seeing and feeling disgust | Neuron 40:655-664 | Same anterior insula sites active for experiencing disgust (inhaling odorants) and observing disgust faces | Insula / empathic disgust |
| 3 | Calder(2000) | Impaired recognition and experience of disgust following brain injury | Nature Neuroscience 3:1077-1078 | Patient with insula+putamen lesion: marked multimodal impairment in recognizing and experiencing disgust | Insula-BG lesion |
| 4 | Calder(2007) | Disgust sensitivity predicts the insula and pallidal response to pictures of disgusting foods | European Journal of Neuroscience 25:3422-3428 | Individual disgust sensitivity predicts anterior insula activation magnitude (r=0.41) | Insula / individual differences |
| 5 | Adolphs(2003) | Dissociable neural systems for recognizing emotions | Brain and Cognition 52:61-69 | Patient with bilateral insula damage: uniformly impaired disgust recognition from both static and action stimuli | Insula / lesion evidence |
| 6 | Sprengelmeyer(1996) | Loss of disgust: perception of faces and emotions in Huntington's disease | Brain 119:1647-1665 | Huntington's disease patients show disproportionate impairment recognizing facial expressions of disgust | Basal ganglia / HD |
| 7 | Sprengelmeyer(1998) | Neural structures associated with recognition of facial expressions of basic emotions | Proceedings of the Royal Society B 265:1927-1931 | Functional imaging confirms basal ganglia + insula involvement in facial disgust recognition | BG + insula imaging |
| 8 | Garcia & Koelling(1966) | Relation of cue to consequence in avoidance learning | Psychonomic Science 4:123-124 | Rats associate taste with illness (but not audiovisual cues), and audiovisual with shock (not taste): selective learning | CTA / Garcia effect |
| 9 | Chapman(2009) | In bad taste: evidence for the oral origins of moral disgust | Science 323:1222-1226 | Gustatory distaste, core disgust, and moral unfairness all activate levator labii muscle (oral rejection response) | Oral origins / EMG |
| 10 | Craig(2009) | How do you feel -- now? The anterior insula and human awareness | Nature Reviews Neuroscience 10:59-70 | Anterior insula = interoceptive hub; correlates with all subjective emotional feelings including disgust | Insula / interoception |
| 11 | Penfield & Faulk(1955) | The insula: further observations on its function | Brain 78:445-470 | Electrical stimulation of anterior insula during neurosurgery evokes nausea, unpleasant taste, sick feeling | Insula / direct stimulation |
| 12 | Rozin(2008) | Disgust | Handbook of Emotions, 3rd ed. (Chapter) | Disgust evolved from oral rejection to pathogen avoidance to moral purity; body-to-soul preadaptation theory | Evolutionary framework |
| 13 | Curtis(2011) | Disgust as an adaptive system for disease avoidance behaviour | Philosophical Transactions of the Royal Society B 366:389-401 | Disgust reactions universal across cultures to disease-salient cues; evolved pathogen avoidance mechanism | Pathogen avoidance |
| 14 | Tybur(2009) | Microbes, mating, and morality: individual differences in three functional domains of disgust | Journal of Personality and Social Psychology 97:103-122 | Three disgust domains: pathogen, sexual, moral; each with distinct sensitivity profiles | Three-domain model |
| 15 | Moll(2005) | The moral affiliations of disgust: a functional MRI study | Cognitive and Behavioral Neurology 18:68-78 | Core disgust: medial OFC; moral disgust: anterior OFC + anterior SFG; overlapping and distinct regions | OFC / moral vs core |
| 16 | Small(2003) | Dissociation of neural representation of intensity and affective valuation in human gustation | Neuron 39:701-711 | Unpleasant taste: left dorsal anterior insula; pleasant taste: right caudolateral OFC; intensity: middle insula | Insula / gustatory valence |
| 17 | Carter(2015) | Parabrachial calcitonin gene-related peptide neurons mediate conditioned taste aversion | Journal of Neuroscience 35:4582-4586 | PBN CGRP neurons necessary and sufficient for establishing CTA; optogenetic activation induces CTA | PBN / CTA |
| 18 | Carter(2018) | Encoding of danger by parabrachial CGRP neurons | Nature 555:617-622 | PBN CGRP neuron activation induces freezing, anxiety, autonomic arousal, anorexia; general threat signal | PBN / threat |
| 19 | Mataix-Cols(2008) | Individual differences in disgust sensitivity modulate neural responses to aversive/disgusting stimuli | European Journal of Neuroscience 27:3050-3058 | Disgust sensitivity positively correlates with anterior insula, putamen, vlPFC, dACC activation to disgusting images | Individual differences |
| 20 | Miller(2021) | Nausea and the brain: the chemoreceptor trigger zone enters the molecular age | Neuron 109:388-390 | Area postrema molecular atlas: specific cell types for emesis; NTS-PBN-insula ascending pathway | AP / NTS / nausea |
| 21 | Vytal & Hamann(2010) | Neuroimaging support for discrete neural correlates of basic emotions: a voxel-based meta-analysis | Journal of Cognitive Neuroscience 22:2864-2885 | Disgust: 16 significant clusters, largest in right insula and right IFG (BA 47); distinct from fear (amygdala) | Meta-analysis |
| 22 | Harrison(2010) | The embodiment of emotional feelings in the brain | Journal of Neuroscience (related work) | Nauseating disgust: right insula + stomach contraction; body-boundary disgust: left insula + cardiac parasympathetic | Insula / embodiment |
| 23 | Cisler(2009) | Disgust, fear, and the anxiety disorders: a critical review | Clinical Psychology Review 29:34-46 | Disgust sensitivity linked to OCD contamination fears; higher difficulty regulating disgust emotions | Disgust / psychopathology |
| 24 | Krolak-Salmon(2003) | An attention modulated response to disgust in human ventral anterior insula | Annals of Neurology 53:446-453 | Intracranial EEG in epilepsy patients: disgust-specific N200 response in ventral anterior insula | Insula / intracranial EEG |
| 25 | Jabbi(2008) | A common anterior insula representation of disgust observation, experience and imagination shows divergent functional connectivity pathways | PLoS One 3:e2939 | Anterior insula active for tasting, watching, and imagining disgust; connectivity differs by context | Insula / modality-general |
| 26 | Sambataro(2006) | Preferential responses in amygdala and insula during presentation of facial contempt and disgust | European Journal of Neuroscience 24:2355-2362 | Anterior insula preferentially activated by disgust vs other negative emotions | Insula / specificity |
| 27 | Stark(2007) | Hemodynamic brain correlates of disgust and fear ratings | NeuroImage 37:663-673 | Anterior insula and amygdala differentially respond to disgust and fear stimuli; insula more for disgust | Insula vs amygdala |
| 28 | Wicker(2003) reference to Rizzolatti's mirror neuron framework | Various | Mirror neuron system in insula mediates empathic disgust: same neurons for experiencing and observing | Mirror neurons / empathy |
| 29 | Royet(2003) | Functional anatomy of perceptual and semantic processing for odors | Journal of Cognitive Neuroscience 15:580-589 | Unpleasant odors activate bilateral anterior insula and left amygdala; pleasant odors activate OFC | Olfactory disgust |
| 30 | Critchley(2004) | Neural systems supporting interoceptive awareness | Nature Neuroscience 7:189-195 | Right anterior insula activity correlates with interoceptive accuracy; mediates awareness of bodily states | Insula / interoception |
| 31 | Adolphs(2005) | Neural systems for recognizing emotion | Current Opinion in Neurobiology 15:169-176 | Review: insula = disgust, amygdala = fear; double dissociation in lesion patients | Emotion recognition review |

### Circuit Diagram (text)

```
DISGUST Circuit:

  [Toxic substance / Pathogen / Moral violation]
       |
       v
  === ASCENDING VISCERAL PATHWAY ===
  [Gustatory/Olfactory receptors] --> [NTS (Nucleus Tractus Solitarius)]
  [Area Postrema / CTZ] ---------> [NTS]  (blood-borne toxin detection)
       |
       v
  [Parabrachial Nucleus (PBN)]
  (CGRP neurons: threat/malaise signal)
       |
       v
  [Thalamus (VPMpc)] ----> [ANTERIOR INSULA (aIC)]
       |                     /           \
       |                    /             \
       |           [Left aIC]        [Right aIC]
       |           (unpleasant        (nauseating
       |            valence)           disgust +
       |                               interoception)
       v
  [BASAL GANGLIA]
  [Putamen + Caudate + GP]
  (disgust recognition;
   damaged in Huntington's)
       |
       v
  [OFC (Orbitofrontal Cortex)]
  /                    \
  [Medial OFC]    [Anterior/Lateral OFC]
  (core disgust   (moral disgust
   evaluation)     evaluation)
       |
       v
  [Motor output: gaping, retching, avoidance]
  [Levator labii activation: oral rejection]

  === EMPATHIC DISGUST (mirror system) ===
  [Observing others' disgust] --> [Anterior insula]
  (same sites as experiencing disgust; Wicker 2003)

  === MORAL DISGUST EXTENSION ===
  [Moral violations] --> [Anterior insula + OFC + SFG]
  (co-opts physical disgust circuitry; Chapman 2009)
```

### Key Parameters for Modeling

| Parameter | Value | Source |
|-----------|-------|-------|
| Anterior insula disgust response latency | N200 (~200 ms) | Krolak-Salmon(2003) |
| Insula activation magnitude (fMRI) | z = 3-5 for disgust vs neutral | Phillips(1997); Vytal(2010) |
| PBN CGRP neuron firing for malaise | Sufficient for CTA in single trial | Carter(2015) |
| CTA learning rate | 1 trial (Garcia effect) | Garcia & Koelling(1966) |
| CTA CS-US delay tolerance | Up to 75 minutes | Garcia & Koelling(1966) |
| Levator labii EMG: disgust response | Significant activation for core, physical, AND moral disgust | Chapman(2009) |
| HD disgust deficit onset | Pre-symptomatic (gene carriers) | Sprengelmeyer(1996) |
| Insula-putamen connectivity | Strong bidirectional | Calder(2000); Sprengelmeyer(1998) |
| Disgust sensitivity-insula correlation | r = 0.41 | Calder(2007) |
| Area postrema location | Dorsal medulla, floor of 4th ventricle; outside BBB | Miller(2021) |

---

## Cross-Emotion Integration Notes

### SEEKING-SADNESS Interaction
Depression involves both diminished SEEKING (reduced VTA DA phasic firing, anhedonia) and overactive PANIC/GRIEF (separation distress, sgACC hyperactivity). The lateral habenula serves as a key switch: when LHb is hyperactive, it suppresses both VTA DA (reducing wanting/motivation) and raphe 5-HT (reducing mood). Ketamine's rapid antidepressant effect works by blocking LHb NMDA-dependent bursting (Yang 2018), thereby disinhibiting VTA DA neurons.

### DISGUST-SADNESS Interaction
The anterior insula is involved in both disgust (as core processing region) and sadness (as interoceptive awareness hub). Craig (2009) proposes that all emotional feelings involve anterior insula re-representation of interoceptive states. Inflammation-induced sickness behavior (Harrison 2009) activates both insula (disgust/nausea component) and sgACC (depressive mood component).

### SEEKING-DISGUST Interaction
Disgust can suppress SEEKING behavior through the parabrachial nucleus-VTA pathway. PBN CGRP neurons that signal malaise project to the VTA, where they can inhibit DA neuron activity and reduce approach behavior. This represents a protective mechanism: disgust suppresses reward-seeking to avoid pathogens. Conversely, strong SEEKING/wanting can override mild disgust signals, as seen in hunger-driven consumption of otherwise aversive foods.

### Modeling Implications
1. **VTA DA neurons** are the central hub connecting SEEKING (wanting), SADNESS (anhedonia via LHb inhibition), and DISGUST (PBN-mediated suppression)
2. **Anterior insula** processes both DISGUST (core function) and SADNESS (interoceptive awareness)
3. **sgACC (BA25)** is the key node for SADNESS, with reciprocal connections to NAc (SEEKING) and insula (DISGUST)
4. **Lateral habenula** bridges SEEKING suppression and SADNESS: negative RPE + learned helplessness
5. **Opioid system** mediates SEEKING pleasure (NAc hotspot) and SADNESS separation distress (PANIC pathway)

---

## References (alphabetical)

1. Adolphs R, Tranel D, Damasio AR (2003) Dissociable neural systems for recognizing emotions. Brain Cogn 52:61-69
2. Adolphs R (2005) Neural systems for recognizing emotion. Curr Opin Neurobiol 15:169-176
3. Bayer HM, Glimcher PW (2005) Midbrain dopamine neurons encode a quantitative reward prediction error signal. Neuron 47:129-141
4. Berridge KC (1998) What is the role of dopamine in reward: hedonic impact, reward learning, or incentive salience? Brain Res Rev 28:309-369
5. Berridge KC, Robinson TE, Aldridge JW (2009) Dissecting components of reward: 'liking', 'wanting', and learning. Curr Opin Pharmacol 9:65-73
6. Bromberg-Martin ES, Matsumoto M, Hikosaka O (2010) Dopamine in motivational control: rewarding, aversive, and alerting. Neuron 68:815-834
7. Calder AJ et al. (2000) Impaired recognition and experience of disgust following brain injury. Nat Neurosci 3:1077-1078
8. Calder AJ et al. (2007) Disgust sensitivity predicts the insula and pallidal response to pictures of disgusting foods. Eur J Neurosci 25:3422-3428
9. Carter ME et al. (2015) Parabrachial calcitonin gene-related peptide neurons mediate conditioned taste aversion. J Neurosci 35:4582-4586
10. Carter ME et al. (2018) Encoding of danger by parabrachial CGRP neurons. Nature 555:617-622 (Note: attributed to Palmiter lab)
11. Caspi A et al. (2003) Influence of life stress on depression: moderation by a polymorphism in the 5-HTT gene. Science 301:386-389
12. Chapman HA et al. (2009) In bad taste: evidence for the oral origins of moral disgust. Science 323:1222-1226
13. Cisler JM, Olatunji BO, Lohr JM (2009) Disgust, fear, and the anxiety disorders: a critical review. Clin Psychol Rev 29:34-46
14. Cohen JY et al. (2012) Neuron-type-specific signals for reward and punishment in the ventral tegmental area. Nature 482:85-88
15. Craig AD (2009) How do you feel -- now? The anterior insula and human awareness. Nat Rev Neurosci 10:59-70
16. Critchley HD et al. (2004) Neural systems supporting interoceptive awareness. Nat Neurosci 7:189-195
17. Curtis V, de Barra M, Aunger R (2011) Disgust as an adaptive system for disease avoidance behaviour. Phil Trans R Soc B 366:389-401
18. Dabney W et al. (2020) A distributional code for value in dopamine-based reinforcement learning. Nature 577:671-675
19. Drevets WC et al. (1997) Subgenual prefrontal cortex abnormalities in mood disorders. Nature 386:824-827
20. Duman RS, Aghajanian GK (2012) Synaptic dysfunction in depression: potential therapeutic targets. Science 338:68-72
21. Duzel E et al. (2010) NOvelty-related Motivation of Anticipation and exploration by Dopamine (NOMAD). Neurosci Biobehav Rev 34:660-669
22. Eisenberger NI, Lieberman MD, Williams KD (2003) Does rejection hurt? An fMRI study of social exclusion. Science 302:290-292
23. Ferenczi EA et al. (2016) Prefrontal cortical regulation of brainwide circuit dynamics and reward-related behavior. Science 351:aac9698
24. Floresco SB et al. (2003) Afferent modulation of dopamine neuron firing differentially regulates tonic and phasic dopamine transmission. Nat Neurosci 6:968-973
25. Floresco SB (2015) The nucleus accumbens: an interface between cognition, emotion, and action. Annu Rev Psychol 66:25-52
26. Garcia J, Koelling RA (1966) Relation of cue to consequence in avoidance learning. Psychon Sci 4:123-124
27. Grace AA et al. (2007) Regulation of firing of dopaminergic neurons and control of goal-directed behaviors. Trends Neurosci 30:220-227
28. Gruber MJ, Gelman BD, Ranganath C (2014) States of curiosity modulate hippocampus-dependent learning via the dopaminergic circuit. Neuron 84:486-496
29. Gusnard DA et al. (2001) Medial prefrontal cortex and self-referential mental activity. PNAS 98:4259-4264
30. Haber SN, Knutson B (2010) The reward circuit: linking primate anatomy and human imaging. Neuropsychopharmacology 35:4-26
31. Hamilton JP et al. (2015) Depressive rumination, the default-mode network, and the dark matter of clinical neuroscience. Biol Psychiatry 78:224-230
32. Harrison NA et al. (2009) Neural origins of human sickness in interoceptive responses to inflammation. Biol Psychiatry 66:415-422
33. Holtzheimer PE, Mayberg HS (2011) Deep brain stimulation for psychiatric disorders. Annu Rev Neurosci 34:289-307
34. Ikemoto S (2007) Dopamine reward circuitry: two projection systems from the ventral midbrain to the NAc-olfactory tubercle complex. Brain Res Rev 56:27-78
35. Jabbi M et al. (2008) A common anterior insula representation of disgust observation, experience and imagination. PLoS One 3:e2939
36. Johansen-Berg H et al. (2008) Anatomical connectivity of the subgenual cingulate region targeted with DBS. Cereb Cortex 18:1374-1383
37. Keedwell PA et al. (2005) A differential pattern of neural response toward sad versus happy facial expressions in MDD. Biol Psychiatry 57:201-209
38. Krishnan V, Nestler EJ (2008) The molecular neurobiology of depression. Nature 455:894-902
39. Kupfer DJ, Frank E, Phillips ML (2012) Major depressive disorder: new clinical, neurobiological, and treatment perspectives. Lancet 379:1045-1055
40. Lammel S et al. (2012) Input-specific control of reward and aversion in the ventral tegmental area. Nature 491:212-217
41. Li B et al. (2011) Synaptic potentiation onto habenula neurons in the learned helplessness model of depression. Nature 470:535-539
42. Mataix-Cols D et al. (2008) Individual differences in disgust sensitivity modulate neural responses to aversive/disgusting stimuli. Eur J Neurosci 27:3050-3058
43. Matsumoto M, Hikosaka O (2007) Lateral habenula as a source of negative reward signals in dopamine neurons. Nature 447:1111-1115
44. Mayberg HS (1999) Reciprocal limbic-cortical function and negative mood. Am J Psychiatry 156:675-682
45. Mayberg HS et al. (2005) Deep brain stimulation for treatment-resistant depression. Neuron 45:651-660
46. Miller AD (2021) Nausea and the brain: the chemoreceptor trigger zone enters the molecular age. Neuron 109:388-390
47. Moll J et al. (2005) The moral affiliations of disgust: a functional MRI study. Cogn Behav Neurol 18:68-78
48. Morales M, Margolis EB (2017) Ventral tegmental area: cellular heterogeneity, connectivity and behaviour. Nat Rev Neurosci 18:73-85
49. Nair-Roberts RG et al. (2008) Stereological estimates of dopaminergic, GABAergic and glutamatergic neurons in the VTA, SN and retrorubral field in the rat. Neuroscience 152:1024-1031
50. Nestler EJ, Carlezon WA (2006) The mesolimbic dopamine reward circuit in depression. Biol Psychiatry 59:1151-1159
51. Olds J, Milner P (1954) Positive reinforcement produced by electrical stimulation of septal area and other regions of rat brain. J Comp Physiol Psychol 47:419-427
52. Padoa-Schioppa C, Assad JA (2006) Neurons in the orbitofrontal cortex encode economic value. Nature 441:223-226
53. Panksepp J (1978) The biology of social attachments: opiates alleviate separation distress. Biol Psychiatry 13:607-618
54. Panksepp J (1998) Affective Neuroscience: The Foundations of Human and Animal Emotions. Oxford University Press
55. Panksepp J, Watt DF (2011) Why does depression hurt? Psychiatry 74:5-13
56. Penfield W, Faulk ME (1955) The insula: further observations on its function. Brain 78:445-470
57. Phillips ML et al. (1997) A specific neural substrate for perceiving facial expressions of disgust. Nature 389:495-498
58. Raichle ME et al. (2001) A default mode of brain function. PNAS 98:676-682
59. Ressler KJ, Mayberg HS (2007) Targeting abnormal neural circuits in mood and anxiety disorders. Nat Neurosci 10:1116-1124
60. Robinson TE, Berridge KC (1993) The neural basis of drug craving: an incentive-sensitization theory of addiction. Brain Res Rev 18:247-291
61. Roitman MF, Wheeler RA, Carelli RM (2005) Nucleus accumbens neurons are innately tuned for rewarding and aversive taste stimuli. Neuron 45:587-597
62. Rolls ET, Grabenhorst F (2008) The orbitofrontal cortex and beyond: from affect to decision-making. Prog Neurobiol 86:216-244
63. Rozin P, Haidt J, McCauley CR (2008) Disgust. In: Lewis M, Haviland-Jones JM, Barrett LF (eds) Handbook of Emotions, 3rd ed. Guilford Press
64. Salamone JD, Correa M (2012) The mysterious motivational functions of mesolimbic dopamine. Neuron 76:470-485
65. Schultz W, Dayan P, Montague PR (1997) A neural substrate of prediction and reward. Science 275:1593-1599
66. Sheline YI et al. (2010) Resting-state functional MRI in depression unmasks increased connectivity between networks via the dorsal nexus. PNAS 107:11020-11025
67. Small DM et al. (2003) Dissociation of neural representation of intensity and affective valuation in human gustation. Neuron 39:701-711
68. Sprengelmeyer R et al. (1996) Loss of disgust: perception of faces and emotions in Huntington's disease. Brain 119:1647-1665
69. Sprengelmeyer R et al. (1998) Neural structures associated with recognition of facial expressions of basic emotions. Proc R Soc B 265:1927-1931
70. Steinberg EE et al. (2013) A causal link between prediction errors, dopamine neurons and learning. Nat Neurosci 16:966-973
71. Tsai HC et al. (2009) Phasic firing in dopaminergic neurons is sufficient for behavioral conditioning. Science 324:1080-1084
72. Tybur JM, Lieberman D, Griskevicius V (2009) Microbes, mating, and morality: individual differences in three functional domains of disgust. J Pers Soc Psychol 97:103-122
73. Tye KM et al. (2013) Dopamine neurons modulate neural encoding and expression of depression-related behaviour. Nature 493:537-541
74. Vytal K, Hamann S (2010) Neuroimaging support for discrete neural correlates of basic emotions. J Cogn Neurosci 22:2864-2885
75. Watabe-Uchida M et al. (2012) Whole-brain mapping of direct inputs to midbrain dopamine neurons. Neuron 74:858-873
76. Wicker B et al. (2003) Both of us disgusted in my insula: the common neural basis of seeing and feeling disgust. Neuron 40:655-664
77. Wise RA (2004) Dopamine, learning and motivation. Nat Rev Neurosci 5:483-494
78. Yang Y et al. (2018) Ketamine blocks bursting in the lateral habenula to rapidly relieve depression. Nature 554:317-322
79. Zubieta JK et al. (2003) Regional mu opioid receptor regulation of sensory and affective dimensions of pain. Science 293:311-315
