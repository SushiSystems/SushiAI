# SushiAI Documentation Manual

This directory contains the technical documentation, design specifications, and guides for SushiAI.

## Getting Started

- [Tutorial](getting_started/TUTORIAL.md): Walkthrough from an empty checkout to a trained model, explaining execution through each pipeline stage.

## Architecture

- [Architecture Overview](architecture/ARCHITECTURE.md): System architecture, execution models, computational graph representation, static memory planning, operator fusion, and D3 dependency layering.

## Guides

- [CLI Guide](guides/CLI_GUIDE.md): Developer command line interface (`sa`) reference for building, testing, profiling, and running models.
- [Integration Guide](guides/INTEGRATION.md): Integration instructions for downstream consumers using CMake and dynamic linking with SushiStack libraries.

## Reference

- [API Reference](reference/API_REFERENCE.md): Overview of primary namespaces, core classes, tensor abstractions, and operator entry points.
- [Glossary](reference/GLOSSARY.md): Definitions of core terms, intermediate representation concepts, and pipeline abbreviations.
- [Known Issues](reference/KNOWN_ISSUES.md): Measured runtime limitations, architectural debt, and hardware compatibility details.
- [Changelog](reference/CHANGELOG.md): History of notable changes, feature additions, and fixes across releases and iterations.
- [Code of Conduct](reference/CODE_OF_CONDUCT.md): Community guidelines and expectations.
- [Security Policy](reference/SECURITY.md): Vulnerability reporting procedures and disclosure processes.

## Design and Roadmaps

- [Design Map](design/README.md): Topic mapping to active architectural design documents.
- [SushiStack Construction & Roadmap](design/ROADMAP.md): Multi-phase implementation roadmap across SushiBLAS and SushiAI.
- [Remaining Work](design/REMAINING_WORK.md): Single backlog tracking open technical debt, planned kernel fusions, and subsystem extensions.

## Contributing and Standards

- [Contributing Guide](CONTRIBUTING.md): Workflow instructions for coding standards, quality checks, test coverage, and pull requests.
- [Documentation Style Guide](DOCUMENTATION_STYLE_GUIDE.md): Structural conventions, naming rules, and prose requirements for all repository documentation.
