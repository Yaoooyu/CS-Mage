# Baseline references and attribution

Use the following citation keys in the CS-Mage paper and release documentation. This file is an attribution record; it does **not** grant permission to redistribute upstream code. Paper links are retained to make final bibliography verification straightforward.

## MMSA framework baselines

| Method | Suggested citation key | Reference | Source |
|---|---|---|---|
| LF-DNN | `poria2018benchmarking` | Poria et al. *Benchmarking Multimodal Sentiment Analysis* (2018). | https://link.springer.com/chapter/10.1007/978-3-319-77116-8_13 |
| TFN | `zadeh2017tfn` | Zadeh et al. *Tensor Fusion Network for Multimodal Sentiment Analysis* (EMNLP 2017). | https://aclanthology.org/D17-1115/ |
| EF-LSTM | `poria2018recognizing` | Poria et al. *Recognizing Emotions in Video Using Multimodal DNN Feature Fusion* (2018). | https://aclanthology.org/W18-3302/ |
| LMF | `liu2018lmf` | Liu et al. *Efficient Low-rank Multimodal Fusion with Modality-Specific Factors* (ACL 2018). | https://aclanthology.org/P18-1209/ |
| MFN | `zadeh2018mfn` | Zadeh et al. *Memory Fusion Network for Multi-View Sequential Learning* (AAAI 2018). | https://arxiv.org/abs/1802.00927 |
| Graph-MFN | `zadeh2018mosei` | Zadeh et al. *Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph* (ACL 2018). | https://aclanthology.org/P18-1208/ |
| MFM | `tsai2019mfm` | Tsai et al. *Learning Factorized Multimodal Representations* (ICLR 2019). | Verify against the MMSA upstream release before final bibliography export. |
| MulT | `tsai2019mult` | Tsai et al. *Multimodal Transformer for Unaligned Multimodal Language Sequences* (ACL 2019). | https://github.com/yaohungt/Multimodal-Transformer |
| MTFN | `yu2020chsims` | Yu et al. *CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotations of Modality* (ACL 2020). | https://aclanthology.org/2020.acl-main.343/ |
| MLF-DNN | `yu2020chsims` | Multi-task late-fusion baseline reported with CH-SIMS. Cite the CH-SIMS paper above. | https://aclanthology.org/2020.acl-main.343/ |
| MLMF | `yu2020chsims` | Multi-task low-rank fusion baseline reported with CH-SIMS. Cite the CH-SIMS paper above. | https://aclanthology.org/2020.acl-main.343/ |
| Self-MM | `yu2021selfmm` | Yu et al. *Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis* (AAAI 2021). | https://ojs.aaai.org/index.php/AAAI/article/view/17289 |
| MMIM | `han2021mmim` | Han et al. *Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis* (EMNLP 2021). | https://arxiv.org/abs/2109.00412 |

The first thirteen methods were called through the MMSA framework in the CS-Mage experiments. Cite both the original method paper and MMSA when the framework itself is described. Upstream MMSA source: https://github.com/thuiar/MMSA .

## Additional baselines

| Method | Suggested citation key | Reference | Source / code status |
|---|---|---|---|
| ConFEDE | `yang2023confede` | Jiuding Yang, Yakun Yu, Di Niu, Weidong Guo, and Yu Xu. *ConFEDE: Contrastive Feature Decomposition for Multimodal Sentiment Analysis*. ACL 2023, pp. 7617–7630. DOI: 10.18653/v1/2023.acl-long.421. | Paper: https://aclanthology.org/2023.acl-long.421/ ; official code used with CS-Mage adaptations. |
| CLGSI | `yang2024clgsi` | Yang Yang, Xunde Dong, and Yupeng Qiang. *CLGSI: A Multimodal Sentiment Analysis Framework based on Contrastive Learning Guided by Sentiment Intensity*. Findings of NAACL 2024, pp. 2099–2110. DOI: 10.18653/v1/2024.findings-naacl.135. | Paper: https://aclanthology.org/2024.findings-naacl.135/ ; code: https://github.com/AZYoung233/CLGSI |
| KuDA | `feng2024kuda` | Xinyu Feng, Yuming Lin, Lihua He, You Li, Liang Chang, and Ya Zhou. *Knowledge-Guided Dynamic Modality Attention Fusion Framework for Multimodal Sentiment Analysis*. Findings of EMNLP 2024. | Paper: https://aclanthology.org/2024.findings-emnlp.865/ ; code: https://github.com/MKMaS-GUET/KuDA |
| CubeMLP | `sun2022cubemlp` | Hao Sun, Hongyi Wang, Jiaqing Liu, Yen-Wei Chen, and Lanfen Lin. *CubeMLP: An MLP-based Model for Multimodal Sentiment Analysis and Depression Estimation*. ACM Multimedia 2022, pp. 3722–3729. DOI: 10.1145/3503161.3548025. | Paper: https://arxiv.org/abs/2207.14087 ; CS-Mage uses a paper-faithful reproduction, not an official repository. |
| EUAR | `gao2024euar` | Zixian Gao, Disen Hu, Xun Jiang, Huimin Lu, Heng Tao Shen, and Xing Xu. *Enhanced Experts with Uncertainty-Aware Routing for Multimodal Sentiment Analysis*. ACM Multimedia 2024, pp. 9650–9659. DOI: 10.1145/3664647.3680949. | Paper: https://doi.org/10.1145/3664647.3680949 ; CS-Mage uses a paper-faithful reproduction because the anonymous official implementation was unavailable. |

## Release wording

> All baselines were evaluated under the same fixed reconstructed five-fold
> protocol and unified metric implementation. Official implementations were
> reused whenever available, while unavailable methods were reproduced
> following their published descriptions.

When preparing the manuscript bibliography, export these entries into the
canonical `.bib` file and verify author spelling, venues, pages, and upstream
licenses against the linked primary source.
