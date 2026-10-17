# CrisisMapping

CrisisMapping is a multimodal crisis-event analysis and mapping project. It combines disaster-related text and images with a trained AI model and a backend API to classify crisis reports and support their use in a mapping system.

The project was developed alongside the research paper **[Multimodal Crisis Event Analysis Mapping System](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12929948)**, published by the [Korean Institute of Information Scientists and Engineers (KIISE)](https://www.dbpia.co.kr/journal/iprdDetail?iprdId=IPRD00010241).

## Repository contents

### `MultiModal Training/`

Contains the multimodal model training code based on the CrisisMMD dataset. It includes two model options:

- **Baseline:** uses the final pooled representation from CLIP's vision encoder.
- **Multi-layer:** combines visual representations from multiple transformer depths using task-specific gating.

The models predict three tasks: informativeness, humanitarian category, and damage severity. See [`MultiModal Training/README.md`](MultiModal%20Training/README.md) for architecture details, dataset preparation, training instructions, results, and limitations.

### `Backend-app/`

Contains the FastAPI backend that integrates the trained CrisisMapping model for disaster-event classification. It provides API endpoints for event submission and retrieval, and supports event updates and simulation functionality. See [`Backend-app/README.md`](Backend-app/README.md) for setup, configuration, endpoints, model integration, deployment, and known limitations.

## Getting started

1. Read the README in `MultiModal Training/` to understand the model, dataset format, checkpoints, and training procedure.
2. Read the README in `Backend-app/` for backend setup, environment configuration, model checkpoint requirements, and API usage.
3. Configure the backend to use a compatible trained checkpoint and provide the required dataset, simulation data, and service configuration as described in its README.

The training and backend components have separate dependencies and configuration. Follow the instructions in each subdirectory rather than assuming one environment setup covers the entire repository.

## Research reference

**Multimodal Crisis Event Analysis Mapping System** — Korean Institute of Information Scientists and Engineers (KIISE).

- Paper: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12929948
- Publisher / institute: https://www.dbpia.co.kr/journal/iprdDetail?iprdId=IPRD00010241
