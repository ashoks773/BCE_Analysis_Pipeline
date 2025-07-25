# Bisulfite Conversion Efficiency Analysis Pipeline

This repository provides a complete and modular pipeline to simulate, analyze, validate, and visualize bisulfite sequencing data for accurate estimation of bisulfite conversion efficiency. It supports multiple measurement strategies, validation metrics, and context-specific analyses.

---

## Repository Structure

- `bisulfite_simulation.py` – Data simulation engine
- `bisulfite_metrics.py` – Efficiency analysis and metric calculations
- `bisulfite_validation.py` – Validation and visualization tools
- `bisulfite_complete_pipeline.py` – Full pipeline integration
- `bisulfite_demo.py` – Demonstration script

---

## Pipeline Components

### 1. Data Simulation (`bisulfite_simulation.py`)
- `BisulfiteSimulator` class with biological realism
- Simulates genomes with custom GC content
- Context-specific methylation (CpG, CHG, CHH)
- Bisulfite conversion with variable efficiency
- Realistic sequencing reads with quality scores and error models

### 2. Metrics and Analysis (`bisulfite_metrics.py`)
- `ConversionEfficiencyAnalyzer` supports:
  1. **Non-CpG method** – CHG/CHH cytosines as proxies
  2. **Lambda DNA method** – Simulates unmethylated spike-ins
  3. **CHH-specific method** – Focus on rarely methylated sites
  4. **Position-specific rates** – Genome-wide patterns
  5. **Context-dependent analysis** – Local sequence effects
  6. **Bootstrap CI** – Confidence interval estimation

### 3. Validation and Visualization (`bisulfite_validation.py`)
- `ValidationFramework` for performance evaluation
- `VisualizationSuite` for high-quality plots
- Metrics include R², MAE, RMSE, correlation, and bias
- Method agreement, cross-validation, and reporting

### 4. Complete Integration (`bisulfite_complete_pipeline.py`)
- `CompleteBisulfitePipeline` class for end-to-end execution
- Automates simulation → analysis → validation → output
- Includes benchmarking and method comparison
- Saves results, plots, and summary reports

### 5. Demonstration (`bisulfite_demo.py`)
- Full working example of pipeline usage
- Step-by-step walkthrough of each module
- Visualizations and interpretation of results
- Best practices and tips

---

## Conversion Efficiency Methods

| Method           | Description                                                   | Notes                                |
|------------------|---------------------------------------------------------------|--------------------------------------|
| **Non-CpG**      | Uses CHG/CHH cytosines (rarely methylated)                    | General-purpose, robust              |
| **Lambda DNA**   | Simulates unmethylated control (spike-in) DNA                 | Good for bias correction             |
| **CHH Context**  | Focuses on CHH sites (<5% methylation in mammals)             | Higher variance, good for validation |
| **Confidence Intervals** | Bootstrap-based uncertainty estimation                | Adds statistical rigor               |

---

## Validation Metrics

- **Accuracy**: R², MAE, RMSE, correlation
- **Bias**: Systematic over/under-estimation
- **Precision**: Within ±1%, ±2%, ±5% thresholds
- **Consistency**: Method agreement (coefficient of variation)
- **Coverage Effect**: Accuracy vs. sequencing depth
- **Range Sensitivity**: Efficiency-range-specific accuracy

### Quality Thresholds

| Category   | R²      | MAE     |
|------------|---------|---------|
| Acceptable | > 0.90  | < 0.02  |
| Good       | > 0.95  | < 0.01  |
| Excellent  | > 0.98  | < 0.005 |

---

## Experimental Recommendations

1. **Primary Method**: Use Non-CpG analysis for routine estimation
2. **Validation**: Combine Lambda DNA spike-ins with other methods
3. **Coverage**: Aim for ≥ 20x sequencing depth
4. **Controls**: Include positive (unmethylated) and negative (methylated)
5. **Replication**: Use technical replicates for sensitive samples

---

## Quick Start

```python
from bisulfite_complete_pipeline import CompleteBisulfitePipeline

# Initialize pipeline
pipeline = CompleteBisulfitePipeline(output_dir="Results_Demo")

# Run the complete analysis
results = pipeline.run_complete_analysis(
    genome_length=10000,
    efficiency_range=(0.90, 0.999),
    n_efficiency_points=8,
    coverage=30
)

## Contact: :raised_back_of_hand:
> [!IMPORTANT]
> For any questions please contact: :point_right: Ashok K. Sharma; ashoks773@gmail.com or compbiosharma@gmail.com

