# Changelog

All notable changes to this project will be documented in this file.

## [2.1] - 2026-01-28

### Features
- **Parallel Processing**: Significant performance boost by parallelizing contour processing using `billiard`.
- **DEM Smoothing**: Added Gaussian smoothing to DEM data to reduce jagged edges.
- **Improved Elevation Sampling**: Implemented windowed reads for elevation checks and improved robustness.
- **SwissTopo Integration**: Integrated Swiss Topo 3D elevation data for high-resolution Swiss terrain.
- **UI Progress Reporting**: Frontend now displays detailed job logs (e.g., "Slicing contours... (Batch 5/10)").
- **User Manual Overhaul**: Completely redesigned the "Manual" modal with better accessibility, keyboard support, and Light Mode compatibility.
- **Downsampling**: Added elevation downscaling options for faster previews.

### Improvements
- **Documentation**: Added High-Level Architecture overview to README and standardized docstrings across the core backend.
- **Frontend Styling**: Refactored sidebar layout and inputs for better usability.
- **Docker**: Optimized `.dockerignore` to exclude large data/media files from build contexts.

### Fixes
- **Modal**: Fixed visibility issues (white-on-white text) in Light Mode.
- **Tests**: Fixed test suite failures related to missing imports and mocking.
- **Robustness**: Improved contour polygon cleaning and error handling during elevation fetching.

## [2.0] - 2026-01-27

### Summary
- Previous stable release.

