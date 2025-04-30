## [Enhancing Clinical Decision Support with Physiological Waveforms — A Multimodal Benchmark in Emergency Care](https://www.sciencedirect.com/science/article/pii/S0010482525005475)

**Accepted by _Computers in Biology and Medicine_**

[![ScienceDirect](https://img.shields.io/badge/ScienceDirect-Read%20the%20paper-green)](https://www.sciencedirect.com/science/article/pii/S0010482525005475)
[![arXiv](https://img.shields.io/badge/arXiv-2407.17856-b31b1b.svg)](https://arxiv.org/abs/2407.17856)


## Clinical Setting

In this study, conducted within the context of an emergency department, we introduce a state-of-the-art biomedical multimodal benchmark. This benchmark is evaluated in two comprehensive settings:

1. **Patient Discharge Diagnoses**: A dataset consisting of 1,428 patient discharge diagnoses.
2. **Patient Deterioration Events**: A dataset consisting of 15 patient deterioration events.

The datasets include various patient data collected within a 90-minute interval upon arrival, such as:
- Demographics
- Biometrics
- Vital parameter trends
- Laboratory value trends
- ECG waveforms

![alt text](https://github.com/AI4HealthUOL/MDS-ED/blob/main/reports/abstract_img.png?style=centerme)


## Comparison to Prior Benchmarks

1. **Comprehensive Size**: MDS-ED ranks first in terms of the number of patients and second in the number of visits in the open-source domain, despite focusing only on the first 1.5 hours of ED arrival.

2. **Features Diversity**: MDS-ED leads in feature modalities, including demographics, biometrics, vital parameter trends, laboratory value trends, and ECG waveforms, making it more extensive than most datasets. Chief complaints and previous medications were excluded due to their unstructured nature and potential bias.

3. **Extensive Range of Target Labels**: MDS-ED offers 1,443 target labels, significantly more than other datasets, which usually have fewer and narrower scope tasks.

4. **Accessibility**: MDS-ED is open-source, promoting further research and collaboration.

![alt text](https://github.com/AI4HealthUOL/MDS-ED/blob/main/reports/related_work.png?style=centerme)


## Proposed Baseline Benchmark

![alt text](https://github.com/AI4HealthUOL/MDS-ED/blob/main/reports/bench.png?style=centerme)


## Conclusions

Overall, we can draw several conclusions:

1. Firstly, the results demonstrate that multimodal models, which integrate diverse data types, offer superior performance in both diagnostic and deterioration tasks (row 4&5 vs. the rest). 

2. Secondly, in the diagnoses task as well as in the deterioration task, the use of ECG raw waveforms instead of ECG features improves the performance in a statistically significant manner (row 4 vs. row 5), finding which is not in line with [ 14 ]. To the best of our knowledge, this is the first statistically robust demonstration of the added value of raw ECG waveform input against ECG features
for clinically relevant prediction tasks such as diagnoses and deterioration prediction. 

3. Thirdly, for the unimodal models, in the deterioration task, the clinical routine data model outperforms ECG features only and ECG waveforms only, however, in the diagnoses task, the ECG waveforms only outperforms the other 2 settings, we hypothesize that for the deterioration task, the clinical routine data apart of including a rich set of clinical features (demographics, biometrics, vital parameters trends, and laboratory values trends) against only an single ECG either in features or waveform, it also includes trends over time which aligns with the task definition of deterioration. Despite this, we believe that a single ECG snapshot can achieve high performances for both tasks, but also we believe that the inclusion of multiple ECGs over time instead of just a single snapshot would allow us to capture more meaningful deterioration and potentially diagnoses trends.


## Reference
```bibtex
@article{ALCARAZ2025110196,
title = {Enhancing clinical decision support with physiological waveforms â€” A multimodal benchmark in emergency care},
journal = {Computers in Biology and Medicine},
volume = {192},
pages = {110196},
year = {2025},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2025.110196},
url = {https://www.sciencedirect.com/science/article/pii/S0010482525005475},
author = {Juan Miguel Lopez Alcaraz and Hjalmar Bouma and Nils Strodthoff},
keywords = {Deep-learning, Emergency department, Machine learning, Medical decision support, Multimodal data, Patient diagnostics, Patient deterioration},
abstract = {Background:
AI-driven prediction algorithms have the potential to enhance emergency medicine by enabling rapid and accurate decision-making regarding patient status and potential deterioration. However, the integration of multimodal data, including raw waveform signals, remains underexplored in clinical decision support.
Methods:
We present a dataset and benchmarking protocol designed to advance multimodal decision support in emergency care. Our models utilize demographics, biometrics, vital signs, laboratory values, and electrocardiogram (ECG) waveforms as inputs to predict both discharge diagnoses and patient deterioration.
Results:
The diagnostic model achieves area under the receiver operating curve (AUROC) scores above 0.8 for 609 out of 1,428 conditions, covering both cardiac (e.g., myocardial infarction) and non-cardiac (e.g., renal disease, diabetes) diagnoses. The deterioration model attains AUROC scores above 0.8 for 14 out of 15 targets, accurately predicting critical events such as cardiac arrest, mechanical ventilation, ICU admission, and mortality.
Conclusions:
Our study highlights the positive impact of incorporating raw waveform data into decision support models, improving predictive performance. By introducing a unique, publicly available dataset and baseline models, we provide a foundation for measurable progress in AI-driven decision support for emergency care.}
}
```
