# RobIA: Robust Instance-aware Continual Test-time Adaptation for Deep Stereo [NeurIPS 2025]
[![Project Website](https://img.shields.io/badge/Project-Website-blue)]() 
- Authors: [Jueun Ko](https://github.com/0ju-un)\*, [Hyewon Park](https://github.com/hhhyyeee)\*, [Hyesong Choi](https://github.com/doihye), Dongbo Min (\* denotes equal contribution)
> 🚨 **Code will be released soon.**


## Abstract
Stereo Depth Estimation in real-world environments poses significant challenges due to dynamic domain shifts, sparse or unreliable supervision, and the high cost of acquiring dense ground-truth labels. While recent Test-Time Adaptation (TTA) methods offer promising solutions, most rely on static target domain assumptions and input-invariant adaptation strategies, limiting their effectiveness under continual shifts. In this paper, we propose RobIA, a novel Robust, Instance-Aware framework for Continual Test-Time Adaptation (CTTA) in stereo depth estimation. RobIA integrates two key components: (1) Attend-and-Excite Mixture-of-Experts (AttEx-MoE), a parameter-efficient module that dynamically routes input to frozen experts via lightweight self-attention mechanism tailored to epipolar geometry, and (2) Robust AdaptBN Teacher, a PEFT-based teacher model that provides dense pseudo-supervision by complementing sparse handcrafted labels. This strategy enables input-specific flexibility, broad supervision coverage, improving generalization under domain shift. Extensive experiments demonstrate that RobIA achieves superior adaptation performance across dynamic target domains while maintaining computational efficiency.

![Method Cover](assets/figure1_1_5.png)
