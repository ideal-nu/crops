<h1 align="center">
    <a href="https://arxiv.org/abs/2610.24209">CROPS: Cosserat-Rod-Based Co-Design of Soft Slender Programmable Material Systems for Biomedical Applications</a>
</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=9wCpPIkAAAAJ&hl=en">Aaditya Chandrasekhar</a>,
  <a href="https://scholar.google.com/citations?user=c1ABHZAAAAAJ&hl=en">Dex Doksoo Lee</a>,
  <a href="https://scholar.google.com/citations?user=lc5z19AAAAAJ&hl=en">Hyunwoo Kwon</a>,
  <a href="https://scholar.google.com/citations?user=xOJ29HUAAAAJ&hl=en">Shicheng Li</a>,
  <a href="https://scholar.google.com/citations?user=mBbNY64AAAAJ&hl=en">Wei Chen</a>
</p>

## Abstract

Slender structures made of stimuli-responsive materials represent an emerging class of programmable material systems, with applications in soft robotics, minimally invasive surgery, wearable electronics, and deployable structures. In biomedical applications, the on-demand design of slender structures, encompassing both their material layout and actuation, is essential to harnessing their large deformation and multiphysics actuation to meet patient-specific design requirements. However, owing to geometric and material nonlinearities, their structure-actuation-response relationship is highly nonintuitive and computationally demanding. This often makes trial-and-error and conventional design approaches challenging for addressing patient-specific needs. To this end, we present a Cosserat-rod-based design optimization framework for the autonomous design of programmable slender structures. Leveraging the Cosserat rod representation, the end-to-end differentiable framework is demonstrated on two case studies: (i) design of swellable stents that morph to the shape of carotid vessels; (ii) co-design of the remnant magnetization and the applied field of magnetically-active catheters to reduce the force exerted on the patient-specific vessel wall during insertion. Both self-contact and external contact with surrounding structures are accounted for. We consider two actuation mechanism, namely the swelling of gel-elastomer composites and the magnetic actuation of hard-magnetic composites. The proposed framework presents an autonomous design platform of soft slender material systems that support large deformations and multiphysics actuation toward personalized biomedical design.

## Citation

```
@article{chandrasekhar2026crops,
author = {Chandrasekhar, Aaditya and Lee, Dex Doksoo and Kwon, Hyunwoo and Li, Shicheng and Chen, Wei},
title = {{CROPS}: Cosserat-Rod-Based Co-Design of Soft Slender Programmable Material Systems for Biomedical Applications},
journal = {arXiv preprint (cs.CE)},
year = {2026},
url = {https://arxiv.org/abs/2610.24209v1},
eprint = {2610.24209v1},
eprinttype = {arxiv}
}
```


## Codebase Organization

*   [`crops/src/`](crops/src): Core source code for the framework.
*   [`crops/notebooks/expts/`](crops/notebooks/expts): Jupyter notebooks that reproduce the experiments in the paper.
*   **Data**: All the raw data can be found on [figshare](https://doi.org/10.6084/m9.figshare.34615671).
