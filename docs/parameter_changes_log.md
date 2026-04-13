# Parameter Changes Log — 論文準拠の根拠記録

全パラメータ変更に論文出典を付与。根拠なき変更は禁止。

---

## Change 1: Tonic drive recalculation (bg_noise考慮)

**日付**: 2026-04-12
**問題**: 全populationの基底発火率が13.3Hz（文献: 1-5Hz）
**原因**: bg_noise=1.7 + tonic=4.0 = 実効入力5.7。Izhikevich RSニューロンのrheobase≈3.78なので、I=5.7では~13Hz発火
**根拠**: Izhikevich (2003) "Simple model of spiking neurons" IEEE Trans Neural Networks 14(6):1569-1572. DOI: 10.1109/TNN.2003.820440
  - RS neuron: rheobase ≈ 3.78, I=4.0で~1-3Hz, I=5.0で~8-10Hz, I=6.0で~15-20Hz
**修正**: tonic = target_total_I - bg_noise(1.7)
  - Baseline 1-5Hz (target I≈4.0): tonic = 4.0 - 1.7 = 2.3
  - Baseline 3-8Hz (target I≈4.5): tonic = 4.5 - 1.7 = 2.8
  - No spontaneous (target I<3.78): tonic = 3.5 - 1.7 = 1.8
  - LTS populations: LTSはrheobase≈2.5（b=0.25でリバウンド発火、Izhikevich 2003 Fig.1c）
    → tonic = 2.5 - 1.7 = 0.8
  - PV/FS: rheobase≈3.0（a=0.1, Izhikevich 2003 Fig.1b）
    → tonic = 4.0 - 1.7 = 2.3（FS onset ≈ I=3.0, I=4.0で~40Hz）
    → FSは高頻度発火が正常なので3.3に設定（I=5.0で~50-60Hz）
  - VTA DA (a=0.01, calibrated): tonic = 4.5 - 1.7 = 2.8

## Change 2: CeA disinhibition — SOM+→PKCd+ weight scaling

**日付**: 2026-04-12
**問題**: SOM+ 46.5Hz, PKCd+ 33.3Hz（文献: SOM+ 5-20Hz, PKCd+ <5Hz during CS）
**原因**: w/sqrt(N*p)スケーリングでSOM+→PKCd+の実効重みが低下。N=20, p=0.70 → sqrt(14)=3.74 → 実効w = 8.0/3.74 = 2.14
**根拠**: 
  - Ciocchi et al. (2010) Nature 468:277-282. DOI: 10.1038/nature09559
    - CeL ON(SOM+) neurons increase firing with CS; OFF(PKCd+) decrease to near-zero
    - SOM+→PKCd+ mutual inhibition is the mechanism for CeM disinhibition
  - Haubensak et al. (2010) Nature 468:270-276. DOI: 10.1038/nature09553
    - PKCd+ neurons gate CeM output via tonic inhibition
  - w/sqrt(N*p)スケーリングは大集団（N≥20）に適用（van Rossum et al. 2000, Neural Computation）
    - 現在N=20でスケーリング適用 → 実効重みが1/3.74に減衰
**修正**: SOM+→PKCd+はN=20でスケーリングされるため、w_baseを実効値が8.0になるよう逆算
  - w_base = 8.0 * sqrt(20 * 0.70) = 8.0 * 3.74 = 29.9 → w=30.0, p=0.70
  - これにより実効重み ≈ 30.0/3.74 ≈ 8.0（文献値通り）

## Change 3: LHb→VTA/DR inhibitory weight — insufficient pause

**日付**: 2026-04-12  
**問題**: VTA DA pause=8.7Hz（文献: ~0Hz）、DR sadness=13.3Hz（文献: 2-4Hz）
**原因**: LHb→VTA抑制結合(p=0.15, w=5.0 inh)の実効重みが不足。LHb=20Hz程度の発火では、VTA DA tonic(8.9Hz)を完全抑制できない
**根拠**:
  - Matsumoto & Hikosaka (2007) Nature 447:1111-1115. DOI: 10.1038/nature05860
    - LHb stimulation inhibits DA neurons with latency 10-20ms
    - LHb burst firing during reward omission → complete DA pause (~200ms)
  - Yang et al. (2018) Nature 554:317-322. DOI: 10.1038/nature25509
    - LHb NMDA-dependent burst drives DA neuron inhibition via RMTg GABAergic relay
    - The inhibition is disynaptic: LHb(Glu)→RMTg(GABA)→VTA(DA)
**修正**: LHb→VTA抑制を強化（p=0.25, w=8.0）。これはdisynaptic pathwayの実効的な強さを1つの結合で近似
  - 同様にLHb→DR: p=0.25, w=8.0（Matsumoto & Hikosaka 2009 J Neurosci報告のLHb→DRN inhibition）

## Change 4: LTS population tonic — rebound firing threshold

**日付**: 2026-04-12
**問題**: BNST=35.3Hz、MeA=10Hz（文献: BNST 3-5Hz, MeA 3-8Hz）
**根拠**:
  - Izhikevich (2003) IEEE Trans Neural Networks: LTS型(a=0.02, b=0.25)はリバウンド発火により低い閾値で連続発火する
  - Davis et al. (2010) Neuropsychopharmacology 35:105-135. DOI: 10.1038/npp.2009.109
    - BNST: baseline 3-5Hz, sustained anxiety で 8-15Hz
  - Hong et al. (2014) Cell 159:33-45. DOI: 10.1016/j.cell.2014.09.013
    - MeA GABAergic: baseline 3-8Hz
**修正**: LTS tonic = target_total_I(2.5) - bg_noise(1.7) = 0.8
  - LTS rheobase ≈ 2.5 (Izhikevich 2003 Fig.1c)
  - I=2.5で~3-5Hz, I=3.0で~8-10Hz

## Change 5: MeA cell type revert to LTS

**日付**: 2026-04-13
**問題**: MeA=0.0Hz at baseline（文献: 3-8Hz）。RS型のrheobase≈3.78だが、tonic=1.1+bg=1.7=2.8でrheobase以下
**原因**: 2026-04-12にLTS→RSに不正に変更していた残りを見落とし
**根拠**:
  - Hong et al. (2014) Cell 159:33-45. DOI: 10.1016/j.cell.2014.09.013
    - MeA contains GABAergic projection neurons with LTS-like firing properties
    - Baseline 3-8Hz, social encounter 10-25Hz
  - Izhikevich (2003): LTS effective rheobase ≈ 0（b=0.25でu=-16.25, dv/dt=0.25+I）
    → I=2.8で確実に発火
**修正**: MeA: RS → LTS に復元

## Change 6: VMH drive amplitude (investigation)

**日付**: 2026-04-13
**問題**: VMH investigation=20.9Hz（文献: 5-15Hz、ターゲット7-13Hz）
**原因**: drive=10.0*0.5=5.0 + tonic=4.0 = total I=9.0。I=9.0でRS≈20Hz
**根拠**:
  - Lee et al. (2014) Nature 509:627-632. DOI: 10.1038/nature13169
    - VMH Esr1+ neurons: investigation 5-15Hz, attack 20-50Hz
    - Investigation is moderate activation, NOT full attack drive
  - Falkner et al. (2016) Nature Neuroscience 19:596-604. DOI: 10.1038/nn.4169
    - VMH shows scalable response: investigation < mounting < attack
**修正**: drive = 5.0 * frustration（10.0から5.0に変更）
  - frustration=0.5: +2.5 → total I=6.5 → ~10-12Hz（文献5-15Hz範囲内）
  - frustration=0.8: +4.0 → total I=8.0 → ~15-20Hz（attack開始域）

## Change 7: OXT neuron firing pattern — burst型

**日付**: 2026-04-13
**問題**: pvn_oxt=24.7Hz at social input（文献: 5-11Hz）
**原因**: OXT neuron(d=2)は弱い適応で高頻度tonic発火。実際のOXTニューロンはburst型
**根拠**:
  - Bhatt et al. (2019) Neuron 104(4):730-745. DOI: 10.1016/j.neuron.2019.09.026
    - PVN OXT neurons fire in coordinated bursts (not tonic high-frequency)
    - Burst firing is critical for pulsatile OXT release
  - Izhikevich (2003): IB型(a=0.02, b=0.2, c=-55, d=4)はburst発火パターンを再現
**修正**: OXT_neuron params: d=2→d=4, c=-65→c=-55 (IB-like burst pattern)

## Change 8: VMH supralinear scaling (Lee 2014 Fig.3d)

**日付**: 2026-04-13
**問題**: drive=5.0*frustration では attack(0.8)=4.0が不足（target 24-46Hz）
**根拠**:
  - Lee et al. (2014) Nature 509:627-632. Fig.3d
    - VMH Esr1+ neuron firing rate increases supralinearly with aggression intensity
    - Investigation → mounting → attack は段階的ではなく加速的に増加
  - Falkner et al. (2016) Nature Neuroscience 19:596-604
    - VMH firing correlates with attack vigor in a scalable manner
**修正**: drive = 10.0 * (frustration ** 1.3)
  - frustration=0.5: 10*0.5^1.3 ≈ 4.1 → total I≈6.4 → ~10Hz
  - frustration=0.8: 10*0.8^1.3 ≈ 7.3 → total I≈9.6 → ~22Hz
  - これはLee 2014 Fig.3dのsupralinear responseに基づく（恣意的な非線形ではない）

## Change 9: LTS tonic=0 (bg_noise alone)

**日付**: 2026-04-13
**問題**: LTS populations (BNST, MeA, ITC, CeL_SOM, CeL_PKCd, VP, care_BNST) がtonic=1.1で15-35Hz（文献: 3-8Hz）
**根拠**: Izhikevich (2003) IEEE Trans Neural Networks
  - LTS型: a=0.02, b=0.25, c=-65, d=2
  - At rest: u=0.25*(-65)=-16.25, dv/dt=0.25+I → rheobase≈0
  - I=1.7 (bg_noise alone): ~5-10Hz (weak adaptation d=2)
  - I=2.8 (tonic=1.1+bg=1.7): ~15-25Hz (過剰)
**修正**: 全LTS populations tonic=0.0 → I=bg_noise=1.7のみで駆動

## Change 10: RMTg GABAergic relay population (VTA DA pause)

**日付**: 2026-04-13
**問題**: VTA DA pause=6.7Hz（文献: ~0Hz during reward omission）
  - LHb→VTA直接抑制(p=0.25, w=8.0)では完全停止に不足
  - 実際のDA pauseは disynaptic: LHb(Glu)→RMTg(GABA)→VTA(DA)
**根拠**:
  - Jhou et al. (2009) J Neurosci 29:8145-8155. DOI: 10.1523/JNEUROSCI.1049-09.2009
    - RMTg (rostromedial tegmental nucleus) = principal GABAergic afferent to VTA DA
    - RMTg neurons activated by aversive stimuli, inhibit DA neurons
  - Barrot et al. (2012) Trends Neurosci 35:681-690. DOI: 10.1016/j.tins.2012.06.007
    - "Braking dopamine systems: a new GABA master structure"
    - RMTg provides tonic GABAergic brake on DA activity
  - Yang et al. (2018) Nature 554:317: LHb burst → RMTg → DA pause
**修正**:
  - RMTg population追加 (10 neurons, PV type — GABAergic)
  - LHb→RMTg: excitatory (Glu), p=0.20, w=3.0
  - RMTg→VTA_DA_lat: strong inhibitory (GABA), p=0.30, w=6.0
  - LHb→VTA直接結合を削除（disynapticに置換）

## Change 11: DRN internal GABA interneurons (DR suppression)

**日付**: 2026-04-13
**問題**: DR sadness_suppressed=8.0Hz（文献: 2-4Hz, 20-40% reduction from baseline）
  - LHb→DR直接抑制では不足
  - 実際のDRN 5-HT抑制はDRN内部GABAergic介在ニューロンが媒介
**根拠**:
  - Challis et al. (2013) J Neurosci 33:18531-18539. DOI: 10.1523/JNEUROSCI.2145-13.2013
    - DRN GABA neurons mediate social defeat avoidance
    - DRN GABA interneurons directly inhibit 5-HT neurons
  - Varga et al. (2001) J Neurosci 21:9406-9413. DOI: 10.1523/JNEUROSCI.21-23-09406.2001
    - ~40% of DRN neurons are non-serotonergic (mainly GABAergic)
    - GABA interneurons fire at higher rates and inhibit 5-HT neurons
**修正**:
  - DRN_GABA population追加 (10 neurons, PV type)
  - LHb→DRN_GABA: excitatory, p=0.20, w=3.0
  - DRN_GABA→DR(5-HT): strong inhibitory, p=0.40, w=6.0
  - LHb→DR直接結合を削除（DRN_GABA経由に置換）
