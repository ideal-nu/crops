![plot](./assets/crops.bmp)

<h1 align="center">
    <a href="https://arxiv.org/abs/2605.19888">CROPS: Cosserat-Rod-Based Co-Design of Soft Slender Programmable Material Systems for Biomedical Applications</a>
</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=9wCpPIkAAAAJ&hl=en">Aaditya Chandrasekhar</a>,
  <a href="https://scholar.google.com/citations?user=c1ABHZAAAAAJ&hl=en">Dex Doksoo Lee</a>,
  <a href="https://scholar.google.com/citations?user=lc5z19AAAAAJ&hl=en">Hyunwoo Kwon</a>,
  <a href="https://scholar.google.com/citations?user=xOJ29HUAAAAJ&hl=en">Shicheng Li</a>,
  <a href="https://scholar.google.com/citations?user=mBbNY64AAAAJ&hl=en">Wei Chen</a>
</p>

## Abstract

Programmable slender structures that deform in response to external stimuli represent a class of emerging material systems, with applications in soft robotics, minimally invasive surgery, wearable electronics, and deployable structures. In biomedical applications, the on-demand design of slender structures, encompassing both their material layout and actuation, is instrumental in leveraging their large deformation and multiphysics actuation for meeting patient-specific design requirements. However, owing to geometric and material nonlinearities, their structure-actuation-response relationship is highly non-intuitive and entails resource-intensive computation. This renders trial-and-error and conventional design approaches largely intractable for addressing patient-specific needs. To this end, we present a Cosserat-rod-based design optimization framework for the autonomous design of programmable slender structures. Capitalizing on the Cosserat rod representation, the end-to-end differentiable framework is demonstrated on two case studies: (i) design of swellable stents that morph to the shape of carotid vessels; (ii) co-design of the remnant magnetization and the applied field of magnetically-active catheters to reduce the force exerted on the patient-specific vessel wall during insertion. As the actuation types, the swelling of gel-elastomer composites and the magnetic actuation of hard-magnetic composites are considered. Additionally, the contact of the rod with itself and its surroundings is accounted for. The proposed framework presents an autonomous design platform for material-structure-actuation co-design of soft slender material systems that support large deformation and multiphysics actuation, with contact addressed.

## Citation

```

@article{chandra2026crops,
  title={{CROPS}: Cosserat-Rod-Based Co-Design of Soft Slender Programmable Material Systems for Biomedical Applications},
  author={Aaditya Chandrasekhar and Dex Doksoo Lee and Hyunwoo Kwon and Shicheng Li and Wei Chen},
  archivePrefix={arXiv},
  eprint={2610.xxxx},
  pages={1--23},
  year={2026},
  primaryClass={cs.CE},
  url={https://arxiv.org/abs/2610.xxxxx}, 
}
```


## Codebase Organization

*   [`crops/src/`](crops/src): Core source code for the framework.
*   [`crops/notebooks/expts/`](crops/notebooks/expts): Jupyter notebooks that reproduce the experiments in the paper.
*   **Data**: All the raw data can be found on [figshare](https://doi.org/10.6084/m9.figshare.34615671).
