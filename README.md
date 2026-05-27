# SENDePSIC

Atomic structure analysis with radial (azimuthal) average & variance profiles. Developed for the ePSIC data processing workflow.

## Installation

To install the package locally in editable mode:
```bash
pip install -e .
```

## Structure

- `sendepsic.radial_profile.radial_profile_analysis`: Class for analyzing radial average & variance profiles from 4D-STEM / EELS-SI data, featuring optimized, in-memory NMF decomposition.
- `sendepsic.feature_extract.feature_extract`: Dimensionality reduction & clustering helper tool.
- `sendepsic.phase.phase_analysis`: Class for crystallographic phase matching.
- `sendepsic.acom`: Utilities for dominant orientation clustering & mapping.
- `sendepsic.synthesis.comprehensive_scientific_synthesis`: Function to generate an integrated scientific summary from RPA, Phase matching, and ACOM data.
- `sendepsic.utils`: Common utilities for data loading, processing, and visualization.
