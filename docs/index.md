---
hide:
  - toc
---

<div class="hero">
  <img src="assets/reionemu-logo.png" alt="reionemu logo" class="hero-logo">
  <p class="hero-kicker">Machine-learning emulator for reionization-era kSZ science</p>
  <h1>A fast emulator for the kinetic Sunyaev-Zel'dovich power spectrum</h1>
  <p class="hero-copy">
    <code>reionemu</code> helps turn simulation outputs into trainable datasets, emulator models,
    and reusable workflows for exploring reionization parameter space without rerunning expensive simulations.
  </p>
  <div class="hero-actions">
    <a class="md-button md-button--primary" href="getting-started/">Get Started</a>
    <a class="md-button" href="api-overview/">Browse API</a>
  </div>
</div>

## What the package covers

<div class="feature-grid">
  <div class="feature-card">
    <h3>Simulation to dataset</h3>
    <p>Condense raw outputs, compute flat-sky power spectra, and assemble training-ready HDF5 datasets.</p>
  </div>
  <div class="feature-card">
    <h3>Training workflows</h3>
    <p>Build dataloaders, train deterministic or MC-dropout emulators, and evaluate validation performance with reusable utilities.</p>
  </div>
  <div class="feature-card">
    <h3>Search and tuning</h3>
    <p>Run Ray Tune experiments to explore architecture and optimizer choices for the deterministic four-parameter emulator.</p>
  </div>
</div>

## Start here

- [Getting Started](getting-started.md) outlines what to include for installation, verification, and contributor setup.
- [API Overview](api-overview.md) gives you a structure for documenting the public surface area.

## Repository layout

- Core package: `src/reionemu/`
- Scripts and HPC workflows: `scripts/`
- Research notebooks: `notebooks/`
- Documentation source: `docs/`
