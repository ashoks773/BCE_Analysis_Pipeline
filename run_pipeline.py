from bisulfite_complete_pipeline import CompleteBisulfitePipeline

pipeline = CompleteBisulfitePipeline(output_dir="Results_demo")
results = pipeline.run_complete_analysis(
    genome_length=10000,
    efficiency_range=(0.90, 0.999),
    n_efficiency_points=8,
    coverage=30
)

