# Questor - Fraud Detection System

A comprehensive fraud detection system combining unstructured text analysis and structured data modeling.

## Features

- **Dual Pipeline Architecture**
  - Unstructured Pipeline: Text document analysis with NER and knowledge graphs
  - Structured Pipeline: Ensemble ML models for tabular data

- **Risk Scoring**
  - Comprehensive risk assessment (0-100 scale)
  - Multi-component risk analysis
  - Risk level categorization (CRITICAL/HIGH/MEDIUM/LOW/MINIMAL)

- **Multiagent Integration**
  - Standardized output format
  - Agent routing recommendations
  - Priority-based processing

## Quick Start

### Unstructured Pipeline
```bash
cd pipelines/unstructured
python calculate_risk_fixed.py --limit 100 --export --batch-name test
```

### Structured Pipeline
```bash
cd pipelines/structured
python structured_pipeline_with_risk.py --export --batch-name test
```

## Documentation

- [Complete System Summary](COMPLETE_SYSTEM_SUMMARY.md)
- [Project Restructuring Plan](PROJECT_RESTRUCTURING_PLAN.md)
- [Unstructured Pipeline Guide](pipelines/unstructured/RISK_SCORING_GUIDE.md)
- [Structured Pipeline Guide](pipelines/structured/RISK_SCORING_GUIDE.md)

## Architecture

See [PROJECT_RESTRUCTURING_PLAN.md](PROJECT_RESTRUCTURING_PLAN.md) for detailed architecture.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up databases
docker-compose up -d

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

## License

[Your License Here]
