# Physics AI for Automotive: A Reference Guide

*A survey of physics-informed and scientific machine learning applied to vehicle aerodynamics, crashworthiness, EV battery thermal management, NVH, tire physics, and autonomous-vehicle physics simulation — seeded for an open-source reference repository.*

> Compiled September 2026. This document synthesizes public research papers, vendor documentation, and industry reporting. Claims are attributed to their sources; where a number or fact could not be independently verified, it is flagged. Treat vendor performance claims (speedups, accuracy figures) as marketing-adjacent unless drawn from a peer-reviewed benchmark.

---

## Table of Contents

1. [Academic Research Landscape](#1-academic-research-landscape)
2. [Model Architectures](#2-model-architectures)
3. [Commercial Products & Startups](#3-commercial-products--startups)
4. [Numerical Methods & Baseline Solvers](#4-numerical-methods--baseline-solvers)
5. [Geometry & Mesh Resources](#5-geometry--mesh-resources)
6. [Datasets & Data Generators](#6-datasets--data-generators)
7. [Industry Standards & Validation Context](#7-industry-standards--validation-context)
8. [Open Challenges & Gaps](#8-open-challenges--gaps)
9. [Roadmap: Building the Reference Hub](#9-roadmap-building-the-reference-hub)
10. [Where this reference meets the PINNeAPPle codebase](#10-where-this-reference-meets-the-pinneapple-codebase)
11. [Contributing](#11-contributing)
12. [Sources](#12-sources)

---

## 1. Academic Research Landscape

Physics AI for automotive sits at the intersection of computational fluid dynamics (CFD), computational solid mechanics (CSM), electrochemistry, and geometric/graph deep learning. A small number of labs and a fast-growing arXiv literature now anchor the field.

### 1.1 Vehicle aerodynamics / CFD surrogates

- **MIT DeCoDE Lab** (Design Computation and Digital Engineering, led by **Faez Ahmed**) is the most visible academic group in this space. In collaboration with the **3D AI Lab at TUM**, the lab produced **DrivAerNet** (Elrefaie et al., published in the *Journal of Mechanical Design*, arXiv:2403.08055) and its successor **DrivAerNet++** (Elrefaie, Morar, Dai, Ahmed; NeurIPS 2024 Datasets & Benchmarks Track, arXiv:2406.09624) — currently the largest open multimodal automotive aerodynamics datasets. DeCoDE also runs a graduate course, *2.S997 AI & ML for Engineering Design*, that feeds this research pipeline.
- **TUM Institute of Aerodynamics and Fluid Mechanics** (Chair of Aerodynamics and Fluid Mechanics, TUM) originated the **DrivAer** reference geometry in 2011 and continues to publish validation and CFD benchmark studies against it, including support for the newer **DrivAerML** high-fidelity dataset (arXiv:2408.11969), produced in partnership with industry (Luminary Cloud, NVIDIA-affiliated groups) using hybrid RANS/wall-modeled LES.
- Groups building **neural operator** and **transformer** surrogates for automotive external aerodynamics have proliferated rapidly through 2025–2026, largely published via arXiv and evaluated on DrivAerNet/DrivAerML/AhmedML/WindsorML: geometry-informed neural operators (GINO) for pressure/wall-shear prediction; **AB-UPT** (Anchored-Branched Universal Physics Transformers, arXiv:2502.09692) jointly modeling surface and volume fields on 100M+ cell meshes; **GAOT** (Geometry-Aware Operator Transformer) combining attentional graph neural operator encoders; a **Geometry-Aware Triplane Field Network** for vehicle aerodynamics (arXiv:2606.07724); and **Transolver-3**, scaling transformer solvers to industrial-scale geometries (arXiv:2602.04940). A benchmarking framework paper (arXiv:2507.10747) and **CarBench** (arXiv:2512.07847) aim to standardize comparison across these architectures.
- Reliability/UQ-focused academic work is emerging on top of these surrogates, e.g. multi-granularity conformal prediction for automotive aerodynamic surrogates on DrivAerML (arXiv:2607.17297).

### 1.2 Crash / crashworthiness surrogates

- Graph neural network (GNN) surrogates for crash mechanics are the dominant architecture class, following the broader "learning to simulate" line of work (encode-process-decode GNNs, MeshGraphNets) adapted to structural mechanics. Recent papers include a GNN surrogate for mesh-based crashworthiness prediction of vehicle panel components (arXiv:2503.17386, also summarized as "a new graph-based surrogate model for rapid prediction of crashworthiness performance of vehicle panel components"), and **"Crash Assessment via Mesh-Based Graph Neural Networks and Physics-Aware Attention"** (arXiv:2605.11784).
- **NVIDIA** has published applied research directly comparing **MeshGraphNet** and **Transolver** (transformer-based) architectures on a Body-in-White crash dataset comprising 150 detailed finite-element (LS-DYNA) simulations, released through the PhysicsNeMo framework and documentation, alongside a companion SAE technical paper, **"Automotive Crash Dynamics Modeling Accelerated with Machine Learning"** (SAE 2026-01-0568; arXiv:2510.15201).
- Uncertainty quantification for crash surrogates is an active niche: a comparative study of Monte Carlo Dropout vs. Deep Ensembles for AI-driven crash surrogates on an open-source bumper-beam benchmark (arXiv:2607.18294), and earlier "nonintrusive uncertainty quantification for automotive crash problems with VPS/Pamcrash" (arXiv:2102.07673) showing that classical (non-ML) UQ pipelines around ESI's PAM-CRASH/VPS solver predate the current ML wave and remain a methodological reference point.

### 1.3 EV battery thermal / electrochemical ML

- Physics-informed neural networks (PINNs) for battery pack thermal management are a growing sub-literature: "Physics-informed Machine Learning for Battery Pack Thermal Management" (arXiv:2411.09915) reports PI-CNN surrogates for temperature-field estimation using heat-conduction-equation-based loss terms, with reported accuracy gains (~15%+) over pure data-driven baselines at equal training data, and separate reviews report PIML reducing drift error by up to 40% and inference cost by up to 250× versus purely data-driven ML in dynamic conditions (claims from review literature; treat as upper-bound, context-dependent figures rather than universal benchmarks).
- Coupled PINN frameworks for **phase-change-material (PCM)** battery pack thermal assessment under high discharge rates (ScienceDirect, 2026) and hard-constrained PINNs for axial temperature estimation in lithium-ion cells (*World Electric Vehicle Journal* 17(5):275) extend this to more realistic EV pack geometries and sparse-sensor settings.
- PINNs are also being applied beyond the cell to the **propulsion motor**: real-time thermal monitoring of EV propulsion motors via PINNs (Springer, 2025 chapter).
- On the data/ML side (not strictly physics-informed but foundational to battery-AI in EVs): NASA's lithium-ion battery aging dataset, the **Oxford Battery Degradation Dataset**, CALCE, and Sandia datasets remain the standard public corpora for state-of-health and remaining-useful-life modeling (see §6).

### 1.4 Venues

- **SAE International** (SAE Mobilus / SAE World Congress and technical paper series) is the primary applied-engineering venue where automotive OEMs and Tier 1s publish ML-for-CAE work (e.g., the NVIDIA/partner SAE 2026-01-0568 crash-ML paper, and an active SAE Mobilus topic collection on NVH).
- **NeurIPS Datasets & Benchmarks Track** has become a landing spot for large automotive ML datasets (DrivAerNet++, 2024).
- Journals: *Journal of Mechanical Design* (ASME), *Journal of Computational Design and Engineering* (Oxford, home to a 2026 survey on generative methods in automotive aerodynamic shape optimization), *Scientific Data* (Nature, battery aging datasets), *ScienceDirect* venues (*Applied Energy*, *Journal of Power Sources*, *Engineering Applications of Artificial Intelligence*), and *Transactions on Machine Learning Research* (TMLR, home of AB-UPT).
- arXiv (cs.LG, physics.flu-dyn, cs.CE) is now the fastest-moving and most complete public record of this field — most 2025–2026 automotive neural-surrogate work appears there first, often months ahead of formal venue publication.

---

## 2. Model Architectures

### 2.1 Neural operators / GNNs for external aerodynamics (drag & flow-field prediction)

- **Graph neural networks on surface/volume meshes** — the RegDGCNN (dynamic graph CNN regression) baseline shipped with DrivAerNet predicts drag directly from 3D meshes, avoiding 2D image rendering or signed-distance-field preprocessing. A 2025 benchmarking paper (arXiv:2504.06699) compares CNN- and GNN-based surrogates on a real-world car external-aerodynamics dataset.
- **Geometry-informed neural operators (GINO)** — a ~230M-parameter model predicting surface pressure and wall shear stress, reporting ~2.1% mean relative error on drag-coefficient prediction on held-out DrivAerNet-class data and claimed to run roughly three orders of magnitude faster than OpenFOAM at inference.
- **DoMINO** — NVIDIA's automotive external-aerodynamics architecture, shipped as a pretrained model within PhysicsNeMo and packaged as an NVIDIA NIM microservice; also the backbone of Luminary Cloud's SHIFT-SUV foundation model (see §3).
- **Anchored-Branched Universal Physics Transformers (AB-UPT)** — jointly models surface and volume CFD fields on meshes exceeding 100M cells; published as *NeuralCFD* / AB-UPT (arXiv:2502.09692, TMLR 10/2025).
- **Point-cloud and triplane representations** — point clouds are increasingly used as an alternative, resolution-agnostic geometry encoding for aerodynamic design search; a 2026 Geometry-Aware Triplane Field Network (arXiv:2606.07724) fits this trend.
- **Reliability layers**: conformal prediction wrapped around neural-operator drag/flow predictions (arXiv:2607.17297) to produce calibrated case-level uncertainty intervals — a prerequisite for using these surrogates in gated engineering decisions rather than only early-stage screening.

### 2.2 Crashworthiness surrogates

- **MeshGraphNet-style encode-process-decode GNNs** with autoregressive time-rollout are the default architecture for learning structural deformation histories from LS-DYNA/explicit-FEM trajectories, following the "Learning to Simulate Complex Physics with Graph Networks" (Sanchez-Gonzalez et al.) lineage adapted to structural mechanics.
- **Transformer-based solvers (Transolver)** are now benchmarked head-to-head against MeshGraphNet on Body-in-White crash datasets inside NVIDIA PhysicsNeMo, alongside a third option, **FIGConvUNet**, for convolution-style mesh processing.
- Component-level surrogates (e.g., bumper beams, panel components) remain the tractable unit for full crash-mode ML today; **full vehicle-level crash surrogates are explicitly called out by Neural Concept's founder as a current performance/accuracy limit** — the company's platform handles individual components but not yet whole-vehicle crash.
- Physics-aware attention mechanisms (arXiv:2605.11784) are an emerging refinement aimed at preserving local structural detail (fold/buckle initiation) that pure message-passing GNNs tend to over-smooth during long autoregressive rollouts.

### 2.3 PINNs for EV battery thermal management

- Physics-informed CNNs embedding the heat-conduction PDE into the training loss for pack-level temperature-field surrogates (arXiv:2411.09915).
- Hard-constraint PINNs enforcing exact boundary/terminal heat-exchange physics while using PDE residuals to infer internal (unsensed) temperature states — enabling accurate estimation from minimal physical sensor coverage, a materially important capability for production BMS (battery management system) cost constraints.
- Coupled multi-physics PINNs for PCM-integrated packs under high discharge rate, extending single-cell PINNs to system-level thermal-electrochemical coupling.

### 2.4 ML for NVH prediction

- NVH ML work is comparatively less mature and more fragmented than aero/crash: published approaches include big-data correlation analysis of NVH from driving data, support-vector-machine models for component-level noise sources (e.g., wiper-motor noise prediction, reported as outperforming traditional CAE for that narrow task), deep-learning-based NVH performance prediction (patent literature, e.g., US 11,556,792), and an end-to-end data-driven framework for intelligent evaluation of automotive noise/vibration performance (ScienceDirect, 2026). Hexagon publishes applied guidance specifically on NVH for EVs (motor whine, gear mesh tonality replacing masked ICE noise as the dominant complaint category). A 2025 arXiv paper on automotive sound-quality psychoacoustic metrics with reproducible AI/ML baselines (arXiv:2509.16901) is a notable step toward benchmarked, reproducible NVH-ML research rather than proprietary point solutions.

### 2.5 Generative models for vehicle shape optimization / design exploration

- GANs applied to aerodynamic shape generation/optimization, and diffusion models applied both to generating vehicle exterior concepts from text/sketch input and, per a 2026 Oxford *JCDE* survey on generative methods in automotive aerodynamic shape optimization, to broader shape-and-field co-generation.
- Optimal-transport-based generative design for shape exploration and solution-field interpolation (arXiv:2511.17111) and LLM-driven parametric shape optimization (arXiv:2412.08072) represent two different 2025–2026 directions: geometric/statistical generative methods vs. language-model-mediated design search.
- Autodesk Research published applied work on a generative AI system to assist conceptual automotive design ("Inspired by AI?").
- On the commercial/applied side, Altair's OptiStruct remains the dominant topology/shape/bead/gauge optimization tool used across automotive structural design, now consolidating under Siemens (see §3); AWS has published a case study on Nifco using generative AI and diffusion models for automotive component design.

### 2.6 Foundation models & simulation stacks for driving-scenario / AV physics

- **NVIDIA DRIVE Sim**, built on **Omniverse**, combines OpenUSD-based scene composition with physically based sensor simulation (camera, radar, lidar) and PhysX-based vehicle dynamics for closed-loop AV testing; the broader **Omniverse Blueprint for real-time digital twins** (CFD-focused) claims up to ~1,200× faster simulation-plus-visualization versus conventional solve-then-render pipelines, per NVIDIA's own November 2024 announcement.
- **LiDARsim** (Uber ATG lineage) and Waabi's lidar-domain-gap research represent the "learn the sensor physics from real data" school — building simulated lidar return models directly from real-world point-cloud captures rather than hand-authored ray-tracing physics, explicitly to close the sim-to-real domain gap (see §8).
- Neural-radiance-field and Gaussian-splatting-based scene reconstruction is increasingly hybridized with geometric/physics simulation for AV sim-to-real transfer (e.g., hybrid Gaussian Splatting + geometric reconstruction for autonomous driving, ScienceDirect 2026).
- Large-language-model-augmented digital twins for "realistic, interactive, and editable simulation for autonomous driving" (arXiv:2507.00319) illustrate the emerging pattern of combining LLM-driven scenario authoring with an underlying physics/rendering engine.

### 2.7 Digital twins for vehicle dynamics

- Real-time multi-physics digital twins for off-road vehicles integrating vehicle dynamics, hydraulics, visual environment, and sensor emulation, explicitly enabling Hardware-in-the-Loop (HIL) and Vehicle-in-the-Loop (VIL) testing.
- Neural-network-augmented digital twins for adaptive automotive suspension control (*Scientific Report*s, 2025), and ML-based digital twins specifically for EV battery modeling (arXiv:2206.08080), showing the digital-twin pattern applied across chassis and powertrain/battery domains, not just aero/crash.
- dSPACE's **ASM Vehicle Dynamics** and IPG's **CarMaker** remain the dominant *physics-based* (non-ML) real-time vehicle-dynamics references against which ML digital twins are typically validated (see §4).

---

## 3. Commercial Products & Startups

| Company | Product(s) | Automotive Physics-AI Focus |
|---|---|---|
| **NVIDIA** | Omniverse, PhysicsNeMo (formerly Modulus), DRIVE Sim, NIM microservices | Platform layer: DoMINO pretrained aero model, CFD/crash surrogate frameworks (PhysicsNeMo), Omniverse Blueprint for real-time CFD digital twins, physically-based AV sensor + vehicle-dynamics simulation (DRIVE Sim). Central "picks-and-shovels" provider that many others (Luminary Cloud, Cadence) build on. |
| **Neural Concept** | Neural Concept Shape (NCS) | Geometric deep learning platform trained on CAE simulation data for near-real-time prediction across CAD geometries; automotive/motorsport/aerospace customers include Airbus, Toyota, Bosch, Mahle, and (per TechCrunch reporting) 4 of 10 F1 teams including Williams Racing. Reported GM collaboration: crash-prediction time reduced from 48 hours to 30 seconds using AI trained on GM crash data, enabling exploration of 10,000+ design variants vs. ~100. NCS runs aero cases in ~20 seconds vs. ~1 hour for traditional CFD (Williams use case). Notably, whole-vehicle crash simulation is cited by the founder as a current capability limit — component-level only today. |
| **Ansys (Synopsys)** | Ansys SimAI | Trains AI models on historical 3D simulation data to predict new design behavior (continuous fields) in minutes across topology variation; automotive applications span external aero, battery cooling, thermal analysis, anti-vibration design. Sumitomo Riko reports >10× simulation acceleration using SimAI. |
| **Siemens (post-Altair acquisition, March 2025)** | Simcenter (unified with Altair), Simcenter PhysicsAI, Simcenter Amesim, Altair OptiStruct | Simcenter PhysicsAI is Siemens' geometric-deep-learning technology converting simulation data into predictive AI models, claimed up to 1,000× faster than solver-based simulation for design evaluation. Magneti Marelli used Simcenter Amesim to cut controls-validation effort ~50% on fuel injection systems. Altair OptiStruct remains the leading topology/shape optimization engine for automotive structures, now integrated into the unified Simcenter portfolio. |
| **Cadence** | Reality Digital Twin Platform, Millennium M1 Supercomputer/Enterprise Multiphysics Platform, Fidelity CFD | GPU-accelerated multiphysics digital-twin platform (built with NVIDIA Omniverse/OpenUSD); Fidelity CFD reports up to 8× simulation-throughput gains (Corvid Technologies, subsonic applications) when paired with GPU compute. Automotive positioned alongside aerospace/defense and energy as a target vertical for digital-twin multiphysics. |
| **Luminary Cloud** | Luminary physics-AI platform, SHIFT Models (SHIFT-SUV) | GPU-native CFD solver plus physics-AI model training; released **SHIFT-SUV**, described as "the world's first Physics AI foundation model and dataset for industrially-relevant automotive SUV aerodynamic analysis," built on NVIDIA PhysicsNeMo's DoMINO architecture, in partnership with **Honda** and **NVIDIA**. Trained on CFD sweeps over an open SUV geometry (~1,000 simulations as of the announcement, targeting 25,000 by year-end). |
| **Monolith AI** | Monolith platform | No-/low-code ML-for-engineering platform used across automotive OEMs for surrogate modeling from existing test/simulation data (general engineering-ML positioning; less automotive-crash-specific than Neural Concept in public reporting). |
| **Applied Intuition** | Simian, ADAS/AV simulation & vehicle-software stack | AV/ADAS simulation, scenario generation, and validation platform; supports importing physics-based vehicle dynamics models. Simian is TCL 3 certified under ISO 26262:2018 Part 8 §11 (TÜV Nord), qualified for ASIL D safety-critical AV/ADAS development — one of relatively few simulation tools with this formal functional-safety tool-confidence certification. |
| **Cognata** | Cognata simulation platform | AV/ADAS simulation combining generative-AI scene/digital-twin generation with real-world + synthetic sensor data; supports GNSS, IMU, camera, lidar, radar simulation; also serves defense, agriculture, mining, warehouse verticals beyond automotive. |
| **Waabi** | Waabi World (simulator), Waabi Driver | Combines learned/generative world models with physics-based sensor and dynamics simulation ("AI and simple physics") for scalable closed-loop AV simulation; publishes lidar sim-to-real / domain-gap research (LiDARsim lineage). |
| **dSPACE** | ASM Vehicle Dynamics, HIL systems | Real-time-capable, physically precise vehicle-dynamics model (drivetrain with elastic shafts, semi-empirical tire models, nonlinear suspension kinematics/compliance, steering, aero) used as the reference physics layer for HIL testing; increasingly co-simulated with ML perception/planning stacks. |
| **IPG Automotive** | CarMaker, TruckMaker, MotorcycleMaker | Physics-based virtual driving-test platform simulating full vehicle + environment; reference tool for reproducible, automated vehicle-dynamics/ADAS regression testing (millions of virtual km), commonly paired with HIL benches. |
| **MORAI** | SIM:Drive | ISO 26262-aligned AV/ADAS simulation with reproducible test reporting supporting ASIL D safety-target verification, including sensor-module (camera) evaluation studies. |
| **Ketryx** | Agentic AI compliance platform | Not a physics simulator — an AI-driven compliance layer targeting ISO 26262 / IEC 61508 evidence generation, relevant to how AI-based simulation tools themselves get qualified in the safety chain. |

**Uncertain / not independently verified in this pass:** "rready" was named in the original research brief as a possible automotive-focused startup; no corroborating public information was found in this research pass, and it is **omitted from the table above pending verification** rather than included speculatively.

---

## 4. Numerical Methods & Baseline Solvers

Physics-AI surrogates in automotive are trained against — and evaluated relative to — an established stack of high-fidelity numerical solvers. Any serious reference repo should treat these as the "ground truth" baselines.

- **External aerodynamics CFD**
  - **Simcenter STAR-CCM+** (Siemens) — general-purpose finite-volume CFD, heavily used for automotive external/internal aero and thermal management.
  - **PowerFLOW** (originally Exa Corporation; acquired by **Dassault Systèmes** in 2017, now part of SIMULIA) — Lattice-Boltzmann-method CFD, the de facto standard for high-fidelity automotive external aerodynamics and aeroacoustics in OEM wind-tunnel-replacement workflows.
  - **OpenFOAM** — open-source finite-volume CFD; the primary engine behind open datasets (AhmedML, WindsorML use OpenFOAM-based hybrid RANS-LES) precisely because it is license-free and reproducible for public benchmark generation.
  - **Cadence Fidelity CFD** — GPU-native CFD/meshing/solving/post-processing platform, positioned for high-throughput automotive/aerospace/energy multiphysics.
- **Crash / impact FEM (explicit dynamics)**
  - **LS-DYNA** (Ansys/formerly LSTC) — the dominant explicit-FEM solver for automotive crashworthiness; the source of nearly all crash-surrogate training data referenced in §1.2/§2.2 (Body-in-White datasets, bumper-beam benchmarks).
  - **Abaqus/Explicit** (Dassault Systèmes SIMULIA) — alternative explicit solver used for impact/crash and occupant-safety analysis.
  - **PAM-CRASH / Virtual Performance Solution (VPS)** — originally ESI Group, now part of **Hexagon** (ESI was acquired by Hexagon in 2023–2024); long-standing crash and occupant-safety simulation standard, also the basis of classical (pre-ML) UQ research (arXiv:2102.07673).
- **Multi-body dynamics**
  - **MSC/Hexagon Adams** — the standard multi-body-dynamics solver for vehicle suspension, steering, and full-vehicle ride/handling simulation, frequently used to generate or validate training trajectories for ML vehicle-dynamics surrogates and digital twins.
  - **dSPACE ASM Vehicle Dynamics** and **IPG CarMaker** — real-time-capable physics vehicle-dynamics models (see §3), which straddle the line between "solver" and "product" — they are the physics engines inside HIL/VIL rigs.
- **Battery simulation**
  - Coupled electrochemical-thermal models following **Doyle-Fuller-Newman (DFN)** / pseudo-2D formulations remain the physics baseline that PINN battery-thermal surrogates are trained or regularized against; commercial implementations exist inside Simcenter, Ansys, and dedicated battery-simulation tools (e.g., COMSOL Battery Design Module), alongside open-source efforts like PyBaMM (Python Battery Mathematical Modelling) as a common open reference implementation for DFN-class models.

---

## 5. Geometry & Mesh Resources

- **DrivAer** — the canonical open reference vehicle geometry from **TUM's Institute of Aerodynamics and Fluid Mechanics** (introduced 2011), combining styling cues from a BMW 3 Series and Audi A4 to bridge the gap between over-simplified bluff-body models and proprietary production geometry. Ships in three body styles — **fastback, estateback, notchback** — each with **smooth and detailed underbody** variants. CAD geometry is published and downloadable, with a companion experimental/numerical validation database (wind-tunnel and CFD results) maintained by TUM.
- **Ahmed body** — the classical simplified bluff-body reference (25°/35° slant-angle variants) used since the 1980s for canonical separated-flow validation; still the default entry point for new CFD-ML methods before scaling to DrivAer-class geometry.
- **SAE reference body** — an older, simplified generic-car reference shape predating DrivAer, still referenced in academic comparisons as the "too simple" end of the fidelity spectrum that motivated DrivAer's creation.
- **DrivAerNet / DrivAerNet++ meshes** — 4,000 (DrivAerNet) and 8,000 (DrivAerNet++) parametrically morphed 3D car meshes derived from the DrivAer baseline, released with full CFD fields, parametric design parameters, segmented part labels, and point-cloud representations (see §6).
- **DrivAerML / DrivAerStar / WindsorML / AhmedML** — further parametrically morphed or geometrically distinct reference-body families released as companion CFD datasets (see §6); collectively these give the community a graduated fidelity ladder from Ahmed body → Windsor body → DrivAer → industrial-grade DrivAerStar geometries.
- **NCAP / crash-test dummy and barrier geometries** — standardized anthropomorphic test devices (Hybrid III, THOR) and deformable/rigid barrier geometries defined by NCAP protocols (Euro NCAP, US NCAP/NHTSA, IIHS) are the geometric standard for crash and occupant-safety simulation; unlike DrivAer, these are governed by regulatory/consortium bodies rather than a single open academic release, and detailed dummy/barrier CAD is typically licensed rather than freely open, which is a notable open-data gap (see §8).

---

## 6. Datasets & Data Generators

### Aerodynamics
- **DrivAerNet** (arXiv:2403.08055) — 4,000 car meshes (~0.5M surface faces), full 3D pressure/velocity/wall-shear fields, models wheels and underbody (uncommon among open datasets); MIT DeCoDE + TUM 3D AI Lab.
- **DrivAerNet++** (arXiv:2406.09624, NeurIPS 2024 D&B) — 8,000 car designs, fastback/notchback/estateback × ICE/EV variants, >39 TB of data: 3D meshes, parametric models, aero coefficients, flow/surface fields, segmented parts, point clouds.
- **DrivAerML** (arXiv:2408.11969) — high-fidelity CFD (hybrid RANS-LES) dataset over parametrically morphed DrivAer geometries, produced with industry partners for drag coefficient, surface pressure, wall shear stress.
- **DrivAerStar** (arXiv:2510.16857) — an "industrial-grade" CFD dataset explicitly aimed at vehicle aerodynamic optimization workflows, positioned as closer to OEM-fidelity than academic datasets.
- **AhmedML** (arXiv:2407.20801) — 500 geometric variations of the Ahmed body, high-fidelity hybrid RANS-LES via OpenFOAM (15–20M cells/case), time-averaged volumetric fields, surface data, force/moment coefficients, CAD, 2D slices; validated against experimental drag/lift data.
- **WindsorML** (arXiv:2407.19320) — high-fidelity CFD dataset built around the Windsor body, another standardized bluff-body geometry, for automotive aerodynamics ML.
- **CarBench** (arXiv:2512.07847) — a comprehensive benchmark suite for neural surrogates on high-fidelity 3D car aerodynamics, intended to standardize cross-paper comparison (a recurring pain point given how many distinct datasets/metrics now exist).

### Crash / structural
- Body-in-White crash dataset (150 LS-DYNA simulations) used in NVIDIA's MeshGraphNet-vs-Transolver comparison and SAE 2026-01-0568.
- Open-source bumper-beam benchmark used for the MC-Dropout vs. Deep Ensemble UQ comparison (arXiv:2607.18294) — notable as one of the few genuinely *open* crash-ML benchmarks (most crash training data, including GM's and other OEMs', is proprietary).
- **NHTSA Vehicle Crash Test Database** (crashstats.nhtsa.dot.gov; data.gov listing; NHTSA Datasets & APIs) — public engineering data from NCAP and compliance crash tests, accessible via public API/scraping; research has assembled derivative sets of ~1,122 publicly available crash tests to support restraint-system ML.
- **NCAP safety-ratings API** (NHTSA SafetyRatings API) — vehicle-variant and star-rating metadata, complementary to raw crash-test engineering data.

### Battery / EV
- **NASA Battery Aging dataset** (e.g., "BatteryAgingARC-FY08Q4," IEEE DataPort) — charge/discharge/EIS cycling data at multiple temperatures; the most widely used public li-ion aging dataset in ML papers.
- **Oxford Battery Degradation Dataset** — 8 Kokam pouch cells (740 mAh, graphite/NMC-LCO chemistry), cycled under simulated real-EV drive-cycle loads at 2C charge, widely paired with NASA data for cross-dataset SOH/RUL benchmarking.
- **CALCE** (University of Maryland) and **Sandia National Laboratories** datasets — additional public multi-chemistry cycling datasets covering varied charge/discharge rates and temperatures.
- A 2024 *Scientific Data* (Nature) paper describes a new **multi-stage lithium-ion battery aging dataset using various experimental design methodologies**, an example of the field actively expanding beyond the legacy NASA/Oxford/CALCE trio.
- A dedicated arXiv survey, "Overview of Publicly Available Degradation Data Sets for Tasks within Prognostics and Health Management" (arXiv:2403.13694), is a useful meta-reference cataloguing this space.

### Synthetic data generation pipelines
- Luminary Cloud's SHIFT-SUV approach — CFD-sweep generation over a single open SUV baseline geometry (~1,000 → target 25,000 simulations) is itself a public example of an industrial-scale synthetic-CFD-data generation pipeline feeding a foundation model.
- NVIDIA PhysicsNeMo's DoMINO training recipe and the Omniverse CFD digital-twin blueprint both document reference pipelines for generating/consuming large parametric CFD sweeps for automotive training data.

---

## 7. Industry Standards & Validation Context

- **NCAP crash standards** (Euro NCAP, US NCAP/NHTSA, IIHS, and regional equivalents in China/Japan/ANCAP) define the regulatory and consumer-rating crash-test protocols (frontal, side, pole, pedestrian, occupant-restraint) that both physical testing and ML crash surrogates must ultimately be validated against. No physics-AI crash surrogate is currently used as a substitute for physical NCAP certification testing — surrogates operate upstream, in early-stage design exploration, not as regulatory evidence.
- **ASME V&V 10 / V&V 10.1** — "Standard for Verification and Validation in Computational Solid Mechanics" (first published 2006, current edition ASME V&V 10-2019, reaffirmed 2025), provides the conceptual framework (verification of numerical fidelity vs. validation against physical reality) and shared vocabulary that crash/structural ML-surrogate credibility arguments are increasingly expected to adopt, even though the standard predates and does not explicitly address neural-network surrogates.
- **SAE simulation-credibility practice** — SAE International (SAE Mobilus technical paper series, SAE World Congress) is where OEMs and suppliers publish applied ML-for-CAE credibility and validation case studies (e.g., SAE 2026-01-0568 on ML-accelerated crash dynamics), functioning as the applied-engineering counterpart to ASME's more formal V&V framework.
- **ISO 26262** ("Road vehicles — Functional safety") governs E/E systems including ADAS/AD software and, critically, **the tools used to develop and verify them** (Part 8, Clause 11: tool confidence level / tool qualification). This is directly relevant to physics-AI: **Applied Intuition's Simian** has achieved **TCL 3** certification under ISO 26262:2018 Part 8 §11 (via TÜV Nord), qualifying it for ASIL D development, and **MORAI's SIM:Drive** provides ISO-26262-aligned reproducible test evidence, including camera-sensor-module evaluation for ASIL D safety-target verification. No equivalent public ISO 26262 tool-qualification claim was found in this research pass for aerodynamics/crash ML surrogates (Neural Concept, Ansys SimAI, NVIDIA PhysicsNeMo) — those tools currently sit in the design-exploration space rather than the certified safety-verification chain, which is an important practical distinction for anyone assessing "AI-in-the-safety-loop" maturity across domains.
- **Implication for AI-based simulation tools generally**: functional-safety tool qualification (ISO 26262) is currently a solved/certified problem mainly in the **AV/ADAS scenario-simulation** commercial segment (Applied Intuition, MORAI), and essentially an **open problem** in the **physics/CAE surrogate** segment (aero, crash, thermal ML surrogates) — none of the CAE-surrogate vendors surveyed here publicly claim ISO 26262 or ASME V&V 10 tool qualification for their ML models as of this research pass.

---

## 8. Open Challenges & Gaps

1. **Generalization across novel shapes.** Aero surrogates (DrivAerNet-class, DoMINO, AB-UPT) are trained on morphs of a small number of parametric baseline bodies (DrivAer, Windsor, Ahmed). EV design paradigms — flush door handles, closed/faux grilles, aggressive underbody flattening, new proportions from skateboard platforms — push geometry outside the training manifold, and it remains an open question how well these surrogates extrapolate versus merely interpolate within the family they were trained on. Luminary/Honda's SHIFT-SUV (trained on a single open SUV baseline, scaling toward 25,000 sims) is a direct attempt to address this by scaling data volume within one vehicle class rather than solving cross-class generalization.
2. **Coupling aero-thermal-battery physics.** EV range, battery degradation, and thermal management are jointly determined by aerodynamic drag (energy consumption), underbody/battery-pack airflow (cooling), and electrochemical-thermal battery behavior — but the current literature largely treats aero surrogates, thermal PINNs, and battery-degradation ML as separate modeling stacks. A jointly trained, multi-physics EV surrogate spanning drag → cooling airflow → pack thermal state → degradation was not found as an established, published capability in this research pass; this is one of the clearest whitespace opportunities for an open-source reference project.
3. **Real-time-capable crash surrogates for early design.** Component-level crash GNN/transformer surrogates (bumper beams, panels) already run in seconds, but whole-vehicle Body-in-White crash response — where load paths, occupant kinematics, and multi-part contact/buckling interact — remains, per Neural Concept's own public statement, an area where current ML "hits the limits of performance." Autoregressive rollout stability (error accumulation over the crash-pulse time series) and preserving local fold/buckle detail without over-smoothing are the specific open technical problems (motivating "physics-aware attention" approaches like arXiv:2605.11784).
4. **Uncertainty quantification for safety-critical predictions.** UQ for crash surrogates (MC Dropout, Deep Ensembles) is nascent and evaluated so far only on small open benchmarks (bumper beams); there is no established practice yet for propagating calibrated uncertainty from an ML crash surrogate into a design decision with the same rigor ASME V&V 10 demands of classical FEM. This is the single biggest blocker to ML crash surrogates moving from "early design screening" to any role adjacent to certification evidence.
5. **Sim-to-real gap for AV physics simulation.** Physically based sensor simulation (camera/lidar/radar rendering, e.g., in DRIVE Sim, CarMaker, Cognata) and learned/data-driven sensor models (LiDARsim, NeRF/Gaussian-splatting scene reconstruction, Waabi's domain-gap research) each close part of the gap but not all of it — texture/reflectivity simplification, sensor noise mismatches, and control/perception latency differences between sim and real remain active, unsolved research areas per the current literature (domain randomization and system identification are the standard mitigations, not solutions).
6. **Fragmentation of NVH-ML.** Unlike aero (DrivAerNet ecosystem) and crash (emerging open benchmarks), NVH lacks a widely adopted open dataset or benchmark; most published work is either narrow (single-component noise sources like wiper motors) or proprietary/patent-form. A reference repo could plausibly become the first to aggregate/standardize open NVH-ML benchmarks, mirroring what DrivAerNet did for aero.
7. **Open, license-free crash and dummy/barrier geometry is scarce.** Unlike DrivAer for aero, there is no equivalent widely adopted open-source, high-fidelity crash-test-relevant vehicle/dummy/barrier geometry family; most usable crash training data is proprietary OEM/Tier-1 data (as with GM's collaboration with Neural Concept), which structurally limits reproducible open crash-ML research relative to aero-ML.
8. **Standards lag.** Neither ASME V&V 10 nor ISO 26262 (outside the AV-scenario-simulation tool-qualification niche) currently has an explicit, published extension addressing neural-network CAE surrogates specifically — credibility arguments today are assembled ad hoc by combining classical V&V language with ML-specific UQ, rather than following an agreed sector standard.

---

## 9. Roadmap: Building the Reference Hub

A phased, realistic plan for an open-source repository to become the leading public reference for "Physics AI for Automotive" over 1–3 years.

### Phase 0 (Months 0–2): Foundation
- Publish this reference document as the repo's `README`/wiki entry point, with the TOC structure above as the site's navigation.
- Stand up a **curated, tagged bibliography** (BibTeX + structured JSON/YAML) of every paper cited here plus ongoing arXiv monitoring (aero, crash, battery, NVH, AV-sim categories), so the repo is queryable, not just readable.
- Build a **dataset registry**: one page per dataset (DrivAerNet(++), DrivAerML, DrivAerStar, AhmedML, WindsorML, NHTSA crash DB, NASA/Oxford/CALCE/Sandia battery sets) with direct download links, license terms, size, and a standardized schema description — this alone would be a genuinely differentiated public resource, since no single page currently aggregates all of them consistently.

### Phase 1 (Months 2–6): Reproducible baselines
- Ship runnable, minimal baseline notebooks/scripts against the **open** datasets only (DrivAerNet/++, AhmedML, WindsorML, NHTSA crash DB subset, NASA/Oxford battery sets) — e.g., a RegDGCNN or GNN drag-prediction baseline, a small MeshGraphNet crash-panel baseline (using the open bumper-beam benchmark), and a PINN battery-thermal baseline reproducing published loss formulations.
- Stand up a lightweight **leaderboard** (drag-coefficient MAE, crash-displacement error, battery SOH RMSE) explicitly modeled on what CarBench and the DrivAerNet benchmarks are individually attempting, but unified across domains rather than siloed to aero.
- Publish a **"who's who" map**: academic labs, companies, tools, standards bodies as a structured, linkable directory (essentially a machine-readable version of §1–§4 of this document).

### Phase 2 (Months 6–14): Community & coverage gaps
- Target the identified whitespace directly: start (or seed a call-for-contribution around) an **open NVH-ML benchmark**, since none currently exists at DrivAerNet's level of adoption — this is the highest-leverage original contribution the repo could make rather than only aggregating others' work.
- Add a **UQ/validation section** operationalizing ASME V&V 10 language for ML surrogates specifically (a "V&V for neural CAE surrogates" checklist/template), directly addressing the standards-lag gap in §8.
- Open contribution guidelines modeled on successful adjacent projects (e.g., how DrivAerNet's GitHub, Papers-with-Code style benchmark tracking, and Hugging Face dataset hosting operate) so external labs can submit datasets/models/benchmarks with review.
- Begin systematic coverage of **commercial tools** as living, versioned comparison pages (capabilities, verticals, certification status) rather than static prose — company positioning changes fast (e.g., the Siemens/Altair merger, ESI's move to Hexagon, Exa's move to Dassault all happened within a few years of each other).

### Phase 3 (Year 2): Depth and credibility infrastructure
- Partner-track outreach to the academic groups already publishing openly (MIT DeCoDE, TUM Chair of Aerodynamics) and to open-tooling projects (PyBaMM for battery, OpenFOAM-based CFD pipelines) to host or mirror reproducible pipelines directly, turning the repo into a genuine aggregation/distribution point rather than a links page.
- Publish a **cross-domain multi-physics EV challenge** (aero drag → underbody cooling airflow → pack thermal → degradation), directly targeting Gap #2 in §8 — even a toy-scale version (e.g., built on DrivAerNet geometries + a public battery model) would be a novel, citable contribution no existing resource currently offers.
- Formalize a **UQ/safety-credibility working group** producing a public draft "credibility checklist for ML-based CAE surrogates in automotive," positioned to eventually interface with SAE Mobilus paper submissions or ASME V&V committee discussion — a genuine standards-adjacent contribution rather than just documentation.

### Phase 4 (Year 2–3): Recognized reference status
- Track adoption signals explicitly (citations of the repo in arXiv papers/SAE papers, dataset registry usage, leaderboard submissions) and iterate the roadmap based on which sections get the most external contribution — the aero section will likely mature fastest given DrivAerNet's head start; crash and NVH will likely need the most direct stewardship given data scarcity.
- Maintain a **living "state of the field" annual report** (similar in spirit to this document) so the repo remains a snapshot-in-time reference that gets refreshed rather than a one-time publication that goes stale — given how fast this literature is moving (multiple new arXiv papers per month across aero/crash/battery), staleness is the single biggest risk to the "leading reference" goal.

---

## 10. Where this reference meets the PINNeAPPle codebase

*Added November 2026.* Everything above describes the field. This section is the other half: the places inside the PINNeAPPle organisation where each topic has something runnable, so a claim here can be checked against a case rather than taken on trust. Links are to live issues and finished experiments, not aspirations.

### Use cases and open work in [`PINNeAPPle`](https://github.com/PINNeAPPle-Labs/PINNeAPPle)

| Topic from this document | Where it lives |
|---|---|
| Section 6 — vehicle surface-pressure datasets | [#1488](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1488) vehicle-engineering surrogate benchmarks recreated: car surface pressure (CarBench) |
| Section 2 — graph networks on structural parts | [#1502](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1502) contact pressure as a field, MeshGraphNet vs Transolver vs GNN |
| Section 4 — crash and impact as a baseline | [#1485](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1485) the crash solver's erosion path creates energy, so a campaign has to stop before the first shock |
| Section 7 — impact and drop validation | [#1503](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1503) drop test: gravity, a floor, the sweep and its surrogate |
| Section 8 — morphing and mission-adaptive aerodynamics | [#1306](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1306) AEROMORPH, a race car that changes its own aerodynamics for the corner it is in |

### Finished experiments in [`PINNeAPPle-Benchmark`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark)

| Experiment | What it measures |
|---|---|
| [`03_bumper_beam_transolver`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/tree/main/experiments/03_bumper_beam_transolver) | A bumper-beam crash surrogate from an OpenRadioss DoE campaign, a Transolver reproduction and an improvement over it, against classical baselines — the closest thing here to the Section 2 architecture comparison, on real crash data. |
| [`09_equal_budget_pinn_fno_deeponet`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/tree/main/experiments/09_equal_budget_pinn_fno_deeponet) | When PINNs beat data-driven surrogates and when they do not, at equal budget, across several PDE families. |

### Related repositories

- [`PINNeAPPle`](https://github.com/PINNeAPPle-Labs/PINNeAPPle) — the library itself: PINNs, neural operators, solvers, geometry, validation.
- [`PINNeAPPle-CFD`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-CFD) — the end-to-end CFD workflow behind the external-aerodynamics rows above.

---

## 11. Contributing

This document is curated, and corrections are the most valuable contribution it receives.

1. Open an issue or a pull request with the specific addition, correction or removal.
2. **Cite a real, checkable source** — a paper, an official repo, a vendor page. Anything not verifiable is marked as such rather than stated as fact.
3. Keep entries factual and neutral; no marketing language for commercial products, including PINNeAPPle's own.
4. When adding a dataset, tool or solver, say what it is, who maintains it and where it lives.
5. Flag what has gone stale — funding rounds, acquisitions and renamed products move quickly in this field.

The full conventions are in [CONTRIBUTING.md](../CONTRIBUTING.md). If you are adding something that has a runnable counterpart in the PINNeAPPle organisation, add a row to Section 10 as well so the claim stays checkable.

---

## 12. Sources

**Datasets & benchmark papers**
- [DrivAerNet (arXiv:2403.08055)](https://arxiv.org/pdf/2403.08055)
- [DrivAerNet++ (arXiv:2406.09624)](https://arxiv.org/abs/2406.09624)
- [DrivAerML (arXiv:2408.11969)](https://arxiv.org/pdf/2408.11969)
- [DrivAerStar (arXiv:2510.16857)](https://arxiv.org/pdf/2510.16857)
- [AhmedML (arXiv:2407.20801)](https://arxiv.org/abs/2407.20801)
- [WindsorML (arXiv:2407.19320)](https://arxiv.org/pdf/2407.19320)
- [CarBench (arXiv:2512.07847)](https://arxiv.org/pdf/2512.07847)
- [A Benchmarking Framework for AI Models in Automotive Aerodynamics (arXiv:2507.10747)](https://arxiv.org/pdf/2507.10747)
- [DrivAerNet GitHub repository](https://github.com/Mohamedelrefaie/DrivAerNet)
- [MIT News — 8,000 car designs (DrivAerNet++)](https://news.mit.edu/2024/design-future-car-with-8000-design-options-1205)
- [MIT DeCoDE Lab — Car Drag Coefficient project](https://decode.mit.edu/projects/dragprediction/)
- [TUM — DrivAer geometry, Institute of Aerodynamics and Fluid Mechanics](https://www.epc.ed.tum.de/en/aer/research-groups/automotive/drivaer/)

**Model architectures**
- [AB-UPT / NeuralCFD (arXiv:2502.09692)](https://arxiv.org/pdf/2502.09692)
- [Geometry-Aware Triplane Field Network (arXiv:2606.07724)](https://arxiv.org/pdf/2606.07724)
- [Transolver-3 (arXiv:2602.04940)](https://arxiv.org/pdf/2602.04940)
- [Multi-Granularity Conformal Prediction for Aerodynamic Surrogates (arXiv:2607.17297)](https://arxiv.org/html/2607.17297v1)
- [Benchmarking CNN/GNN surrogates on real-world car aerodynamics (arXiv:2504.06699)](https://arxiv.org/pdf/2504.06699)
- [GNN surrogate for crashworthiness of vehicle panels (arXiv:2503.17386)](https://arxiv.org/pdf/2503.17386)
- [Crash Assessment via Mesh-Based GNNs and Physics-Aware Attention (arXiv:2605.11784)](https://arxiv.org/html/2605.11784)
- [Automotive Crash Dynamics Modeling Accelerated with ML (arXiv:2510.15201 / SAE 2026-01-0568)](https://arxiv.org/abs/2510.15201)
- [NVIDIA PhysicsNeMo — Crash Dynamics example](https://archive.docs.nvidia.com/physicsnemo/25.11/physicsnemo/examples/structural_mechanics/crash/README.html)
- [Uncertainty Quantification for AI Crash Surrogates (arXiv:2607.18294)](https://arxiv.org/abs/2607.18294)
- [Nonintrusive UQ for automotive crash with VPS/Pamcrash (arXiv:2102.07673)](https://arxiv.org/pdf/2102.07673)
- [Physics-informed ML for Battery Pack Thermal Management (arXiv:2411.09915)](https://arxiv.org/pdf/2411.09915)
- [Synergizing physics and ML for battery management (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11413374/)
- [PINNs for EV propulsion motor thermal monitoring (Springer)](https://link.springer.com/chapter/10.1007/978-3-032-31144-3_8)
- [Automotive Sound Quality for EVs: psychoacoustic ML baselines (arXiv:2509.16901)](https://arxiv.org/pdf/2509.16901)
- [Survey on generative methods in automotive aerodynamic shape optimization (Oxford JCDE)](https://academic.oup.com/jcde/article/13/1/538/8408453)
- [Optimal Transport for Generative Shape Exploration (arXiv:2511.17111)](https://arxiv.org/pdf/2511.17111)
- [LLMs for Parametric Shape Optimization (arXiv:2412.08072)](https://arxiv.org/pdf/2412.08072)
- [Autodesk Research — Generative AI for conceptual automotive design](https://www.research.autodesk.com/publications/generative-ai-automotive-design/)
- [NVIDIA Omniverse Real-Time Physics Digital Twins announcement](https://nvidianews.nvidia.com/news/nvidia-announces-omniverse-real-time-physics-digital-twins-with-industry-software-leaders)
- [NVIDIA Omniverse Blueprint — Digital Twins for Fluid Simulation (GitHub)](https://github.com/NVIDIA-Omniverse-blueprints/digital-twins-for-fluid-simulation)
- [Waabi — Towards zero domain gap: realistic LiDAR simulation](https://waabi.ai/research/lidar-dg)
- [LiDARsim (arXiv:2006.09348)](https://arxiv.org/pdf/2006.09348)
- [LLM-augmented digital twins for AV simulation (arXiv:2507.00319)](https://arxiv.org/html/2507.00319v1)
- [ML-based digital twin for EV battery modeling (arXiv:2206.08080)](https://arxiv.org/pdf/2206.08080)
- [Real-time multi-physics digital twin, off-road vehicle (MDPI Machines)](https://doi.org/10.3390/machines14010128)

**Commercial products & companies**
- [NVIDIA PhysicsNeMo developer page](https://developer.nvidia.com/physicsnemo)
- [NVIDIA CAE solutions](https://www.nvidia.com/en-us/solutions/cae/)
- [Neural Concept — TechCrunch on Formula 1 aerodynamic AI](https://techcrunch.com/2024/04/14/how-neural-concepts-aerodynamic-ai-is-shaping-formula-one/)
- [Neural Concept — company site](https://www.neuralconcept.com/)
- [Ansys SimAI (Synopsys)](https://ansys.synopsys.com/products/ai/simai)
- [Ansys SimAI unveiling (Nasdaq)](https://www.nasdaq.com/articles/ansys-anss-unveils-ansys-simai-to-enhance-simulation-dynamics)
- [Sumitomo Riko + Ansys AI 10x simulation acceleration](https://seekingalpha.com/pr/20259061-sumitomo-riko-leverages-ansys-ai-technology-to-accelerate-simulation-over-10x-for-automotive)
- [Siemens Simcenter AI/ML](https://www.siemens.com/en-us/products/simcenter/ai-ml-innovation/)
- [Siemens unifies AI and Altair technologies in Simcenter](https://roboticsandautomationnews.com/2026/07/29/siemens-unifies-simulation-portfolio-with-ai-powered-simcenter-release/103691/)
- [Cadence Reality Digital Twin Platform](https://www.cadence.com/en_US/home/tools/reality-digital-twin.html)
- [Cadence Millennium Supercomputer / digital twins (Forbes)](https://www.forbes.com/sites/karlfreund/2024/02/01/cadence-supercomputer-appliance-enables-full-scale-digital-twins/)
- [Luminary Cloud / Honda / NVIDIA SHIFT-SUV announcement (Engineering.com)](https://www.engineering.com/luminary-cloud-releases-open-source-physics-ai-model-with-honda-nvidia/)
- [Applied Intuition — automotive/AV simulation](https://www.appliedintuition.com/industries/automotive)
- [Applied Intuition — Simian ISO 26262 certification](https://www.appliedintuition.com/press-releases/iso-26262)
- [Cognata](https://www.cbinsights.com/compare/applied-intuition-vs-cognata)
- [Waabi Series B / simulator (TechCrunch)](https://techcrunch.com/2022/02/09/waabis-new-simulator-could-scale-autonomous-vehicle-tech-faster/)
- [dSPACE ASM Vehicle Dynamics Models](https://www.dspace.com/en/inc/home/products/sw/automotive_simulation_models/produkte_asm/vehicle_dynamics_models.cfm)
- [IPG Automotive — Vehicle Dynamics / CarMaker](https://ipg-automotive.com/areas-of-application/vehicle-dynamics/)
- [MORAI — ISO 26262](https://www.morai.ai/iso26262)
- [Ketryx — ISO 26262 / IEC 61508 compliance](https://www.ketryx.com/blog/navigating-iso-26262-and-iec-61508-3-functional-safety-standards)

**Numerical solvers**
- [PowerFLOW (Dassault/Technia)](https://www.technia.com/en/advanced-simulation/software/powerflow/)
- [Exa Corporation — Wikipedia](https://en.wikipedia.org/wiki/Exa_Corporation)
- [Pam-Crash — Wikipedia](https://en.wikipedia.org/wiki/Pam-Crash)
- [ESI Virtual Performance Solution](https://myesi.esi-group.com/downloads/software-downloads/virtual-performance-solution-2020.5)

**Standards & validation**
- [ASME V&V 10-2019](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-solid-mechanics)
- [ASME V&V 10.1-2012 (R2022)](https://www.asme.org/codes-standards/find-codes-standards/an-illustration-of-the-concepts-of-verification-and-validation-in-computational-solid-mechanics)
- [ASME VVUQ overview](https://www.asme.org/codes-standards/publications-information/verification-validation-uncertainty)
- [Synopsys — What is ISO 26262?](https://www.synopsys.com/glossary/what-is-iso-26262.html)

**Crash / battery data**
- [NHTSA Datasets and APIs](https://www.nhtsa.gov/nhtsa-datasets-and-apis)
- [NHTSA Crash Stats / NCSA publications](https://crashstats.nhtsa.dot.gov/)
- [DOT Vehicle Crash Test Database (data.gov)](https://catalog.data.gov/dataset/vehicle-crash-test-database)
- [NASA Li-ion Battery Dataset (IEEE DataPort)](https://ieee-dataport.org/documents/nasa-lithium-ion-battery-dataset)
- [Multi-stage li-ion battery aging dataset (Scientific Data / Nature)](https://www.nature.com/articles/s41597-024-03859-z)
- [Overview of public PHM degradation datasets (arXiv:2403.13694)](https://arxiv.org/pdf/2403.13694)

**NVH**
- [Hexagon — NVH for electric vehicles](https://hexagon.com/resources/resource-library/noise-vibration-and-harshness-nvh-for-electric-vehicles)
- [SAE Mobilus — NVH topic collection](https://saemobilus.sae.org/topics/noise-vibration-and-harshness-nvh)

---

*Compiled by research pass on 2026-09-16. Sections 10 and 11 were added in November 2026 to tie each topic to the runnable part of the PINNeAPPle organisation. This is a living-document seed: several 2026-dated arXiv preprints cited above are recent and may not yet be peer-reviewed; treat vendor-reported performance multipliers (speedups, accuracy percentages) as claims to be independently reproduced, not settled benchmarks. Contributions, corrections, and citation updates are the intended lifeblood of the repository this document seeds.*
