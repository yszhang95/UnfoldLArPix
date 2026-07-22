PowerPoint Structure for “Ionization Charge Unfolding at Zero-Suppressed Liquid Argon Time Projection Chamber”

Based on the uploaded paper  ￼

⸻

Slide 1 — Title Slide

Ionization Charge Unfolding at Zero-Suppressed Liquid Argon Time Projection Chamber

* Author: An Author
* Institution: Brookhaven National Laboratory
* Research Area:
    * Liquid Argon Time Projection Chamber (LArTPC)
    * Signal Processing
    * Charge Reconstruction
    * ND-LAr Detector

⸻

Slide 2 — Motivation and Background

Motivation

* The DUNE Near Detector (ND-LAr) uses:
    * Pixelated readout
    * Zero-suppressed electronics
* Traditional signal processing assumes:
    * Full waveform sampling
    * Linear and time-invariant systems
* Zero suppression breaks these assumptions.

Main Challenge

* Missing waveform information due to threshold-based triggering
* Long induction signals spread across neighboring pixels
* Ghost hits and pre-triggering effects degrade reconstruction quality

Goal of This Work

Develop a physics-guided waveform recovery and deconvolution framework for accurate ionization charge unfolding in zero-suppressed detectors.

⸻

Slide 3 — ND-LAr Readout Characteristics

ND-LAr Readout System

* Self-triggering architecture
* Front-end electronics remain dormant until threshold crossing
* ADC records charge only after discriminator trigger

Key Features

* Threshold-based acquisition
* Burst mode support
* Front-end reset after digitization

Consequences

* Partial waveform loss
* Timing ambiguity
* Missing negative induction signals
* Charge bias due to reset mechanism

Importance

Precise charge reconstruction is essential for:

* Neutrino energy reconstruction
* Vertex reconstruction
* Flavor tagging
* Event topology analysis

⸻

Slide 4 — Traditional Signal Processing

Conventional Signal Processing Model

m(t) = r(t) * s(t) + n(t)

Where:

* m(t): measured waveform
* r(t): detector + electronics response
* s(t): ionization signal
* n(t): noise

Standard Deconvolution

Performed in frequency space using FFT:

\hat{S}(\omega) = \frac{M(\omega)}{R(\omega)}

Standard Techniques

* Gaussian filtering
* Wiener-inspired filtering
* ROI selection
* Noise suppression

Limitation

Traditional deconvolution fails because zero suppression destroys waveform continuity and linearity.

⸻

Slide 5 — Overall Algorithm Framework

Proposed Signal Processing Pipeline

1. Recovery of the full waveform
2. Waveform compensation using physics templates
3. 3D deconvolution
4. Active voxel identification

Core Idea

Infer missing waveform segments using detector-response templates and recover an approximate continuous waveform before FFT deconvolution.

⸻

ALGORITHM SECTION

⸻

Slide 6 — Algorithm Stage 1: Waveform Recovery

Objective

Recover missing waveform information from zero-suppressed burst sequences.

First Pass: Sequence Merging

* Consecutive hits are grouped together
* Uses ADC-HOLD-DELAY criterion
* Threshold-subtracted charge records are appended

Second Pass: Template-Based Compensation

* Detector-response templates fill gaps between sequences
* Transition point determined from threshold crossing
* Template scan identifies missing waveform segments

Important Considerations

* Trigger alignment ambiguity
* Slow digitization effects
* Approximate rather than exact waveform recovery

⸻

Slide 7 — Algorithm Stage 1: Time Alignment Methods

Two Alignment Strategies

Method 1 — Global Binning

* Round signals to nearest global time bin
* Simpler implementation
* Dead-time compensation applied

Method 2 — Fractional FFT Phase Shift

* Apply phase factors in Fourier space
* Preserve sub-bin timing information
* No explicit dead-time compensation

Comparison Goal

Study timing precision versus computational complexity.

⸻

Slide 8 — Algorithm Stage 2: 3D Deconvolution

Deconvolution Workflow

* Recovered waveforms assembled into a 3D tensor
* Deconvolution performed using:
    * 3D field response
    * Gaussian filtering

Computational Technique

* FFT-based deconvolution
* Linearithmic computational scaling

Advantages

* Corrects detector response smearing
* Recovers ionization charge distribution
* Enables spatial charge unfolding

Planned Improvement

* Block-wise deconvolution optimization

⸻

Slide 9 — Algorithm Stage 3: Active Voxel Identification

Current Strategy

* Apply Gaussian filter
* Select voxels above threshold

Current Threshold

500\ e^{-}

Corresponding approximately to front-end noise standard deviation.

Ongoing Development

Wiener-inspired frequency filter:

F = \exp \left[-\left(\frac{\omega}{a}\right)^b \right]

Goal

Improve signal/noise discrimination and voxel selection accuracy.

⸻

PERFORMANCE SECTION

⸻

Slide 10 — Simulation Framework

Simulation Setup

* Detector geometry:
    * ProtoDUNE-ND / 2×2 experiment
* Input particles:
    * Positron particle gun
* Simulation package:
    * GPU-accelerated TRED framework

Evaluation Metrics

* Charge residuals
* Noise robustness
* Threshold dependence
* Template performance

⸻

Slide 11 — Performance: Template Studies

Templates Evaluated

1. Center-pixel field response
2. Average collection-pixel response
3. Collection + neighboring-pixel response
4. Hybrid response model
5. Filtered-template study (ongoing)

Key Observation

Including neighboring-pixel induction improves waveform recovery and reduces ghost-hit effects.

Ongoing Work

Optimization of adaptive and filtered templates.

⸻

Slide 12 — Performance: Filter Studies

Spatial Filtering

* Pixel-axis Gaussian filter:
    * \sigma_p = 0.8 pitch

Temporal Filtering

Tested multiple shaping widths:

* 1.6 μs
* Several larger μs configurations

Goal

Balance between:

* Noise suppression
* Charge resolution
* Timing preservation

⸻

Slide 13 — Performance: Threshold Dependence

Threshold Configurations Tested

* No threshold
* 1000 e^{-}
* 5000 e^{-}

Observations

* Lower threshold preserves more waveform information
* High thresholds increase waveform ambiguity
* Slow digitization becomes more problematic at high thresholds

Conclusion

Low-threshold operation significantly improves deconvolution quality.

⸻

Slide 14 — Performance: Noise Dependence

Noise Model

* Based on LArPix-v2 electronics

Comparison

* With simulated noise
* Without simulated noise

Result

Noise is a subleading effect under current setup conditions.

Interpretation

Waveform truncation and threshold effects dominate reconstruction uncertainty.

⸻

Slide 15 — Reconstruction Performance Summary

Reconstruction Quality

* Residual width:
    * Below 500 electrons

Achievements

* Successful charge unfolding
* Reduced ghost-hit contamination
* Improved ionization charge estimation

Significance

Enables precise detector-resolution quantification and truth-level comparisons.

⸻

DISCUSSION SECTION

⸻

Slide 16 — Discussion: What Has Been Achieved

Completed Contributions

* Developed waveform recovery framework
* Extended deconvolution to zero-suppressed detectors
* Demonstrated FFT-based 3D unfolding
* Evaluated thresholds, templates, and filters
* Integrated simulation truth bookkeeping

Impact

Provides a practical signal-processing pipeline for ND-LAr pixel detectors.

⸻

Slide 17 — Discussion: What Has Not Been Fully Solved

Current Limitations

* Exact waveform start-time recovery remains ambiguous
* Dead-time effects not fully resolved
* Neighbor-induced negative signals remain challenging
* Deconvolution currently assumes simplified templates
* Wiener-inspired voxel selection still under development

Missing Evaluations

* Full CPU vs GPU benchmark
* Saturation studies
* Large-scale detector validation

⸻

Slide 18 — Discussion: Planned Future Work

Ongoing and Planned Developments

Signal Processing

* Adaptive Wiener-inspired filtering
* Filtered-template optimization
* Improved timing reconstruction

Hardware Studies

* GPU acceleration benchmarking
* Burst-mode optimization
* Threshold uniformity studies

Detector Design

* Shield-grid signal shaping
* Suppression of long-range induction
* Reduction of ghost hits

Long-Term Goal

Enable precision charge reconstruction for large-scale neutrino oscillation measurements.

⸻

Slide 19 — Shield Grid Study

Concept

Introduce a shield grid to reshape induction signals.

Expected Effects

* Sharper collection-pixel signals
* Reduced neighboring induction
* Shorter effective waveform duration

Benefits

* Cleaner imaging
* Fewer ghost hits
* Smaller charge residual width

Importance

Potential detector-design improvement for future ND-LAr systems.

⸻

Slide 20 — Conclusions

Summary

This work presents a signal-processing framework for zero-suppressed liquid argon detectors.

Main Contributions

* Physics-guided waveform compensation
* FFT-based 3D deconvolution
* Active voxel reconstruction
* Charge-resolution evaluation

Final Outcome

The method enables accurate ionization charge unfolding despite incomplete waveform acquisition.

Broader Impact

Supports precision neutrino reconstruction and future machine-learning-based event reconstruction workflows.
