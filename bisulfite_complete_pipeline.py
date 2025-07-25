#!/usr/bin/env python3
"""
Bisulfite Conversion Efficiency Analysis Pipeline
Part 4: Complete Integration and Main Pipeline

This module integrates all components and provides the main execution pipeline
for comprehensive bisulfite conversion efficiency analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules (in practice, these would be separate files)
from bisulfite_simulation import BisulfiteSimulator
from bisulfite_metrics import ConversionEfficiencyAnalyzer, calculate_advanced_metrics
from bisulfite_validation import ValidationFramework, VisualizationSuite

class CompleteBisulfitePipeline:
    """
    Complete integrated pipeline for bisulfite conversion efficiency analysis
    """
    
    def __init__(self, output_dir="bisulfite_analysis_results", random_seed=42):
        """
        Initialize the complete pipeline
        
        Args:
            output_dir (str): Directory to save all results
            random_seed (int): Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set random seeds
        np.random.seed(random_seed)
        
        # Initialize components
        self.simulator = None
        self.analyzer = None
        self.validator = ValidationFramework()
        self.visualizer = VisualizationSuite()
        
        # Results storage
        self.simulation_results = {}
        self.analysis_results = {}
        self.validation_results = {}
        self.pipeline_metadata = {
            'start_time': datetime.now().isoformat(),
            'random_seed': random_seed,
            'version': '1.0.0'
        }
        
        print(f"Initialized Bisulfite Analysis Pipeline v{self.pipeline_metadata['version']}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Random seed: {random_seed}")
    
    def run_complete_analysis(self, 
                            genome_length=10000,
                            read_length=150,
                            coverage=30,
                            efficiency_range=(0.90, 0.999),
                            n_efficiency_points=8,
                            gc_content=0.42,
                            methylation_rates=None,
                            error_rate=0.001):
        """
        Run the complete analysis pipeline
        
        Args:
            genome_length (int): Length of simulated genome
            read_length (int): Sequencing read length
            coverage (int): Sequencing coverage depth
            efficiency_range (tuple): Range of conversion efficiencies to test
            n_efficiency_points (int): Number of efficiency points to test
            gc_content (float): GC content of simulated genome
            methylation_rates (dict): Custom methylation rates by context
            error_rate (float): Sequencing error rate
            
        Returns:
            dict: Complete analysis results
        """
        print("\n" + "="*80)
        print("STARTING COMPLETE BISULFITE CONVERSION EFFICIENCY ANALYSIS")
        print("="*80)
        
        # Step 1: Setup and Configuration
        print("\n1. CONFIGURATION AND SETUP")
        print("-" * 40)
        
        config = self._setup_configuration(
            genome_length, read_length, coverage, efficiency_range,
            n_efficiency_points, gc_content, methylation_rates, error_rate
        )
        
        # Step 2: Generate Ground Truth Data
        print("\n2. GROUND TRUTH DATA GENERATION")
        print("-" * 40)
        
        ground_truth = self._generate_ground_truth_data(config)
        
        # Step 3: Simulate Bisulfite Sequencing Data
        print("\n3. BISULFITE SEQUENCING SIMULATION")
        print("-" * 40)
        
        simulation_results = self._simulate_bisulfite_data(ground_truth, config)
        
        # Step 4: Analyze Conversion Efficiency
        print("\n4. CONVERSION EFFICIENCY ANALYSIS")
        print("-" * 40)
        
        analysis_results = self._analyze_conversion_efficiency(simulation_results, ground_truth)
        
        # Step 5: Validate Results
        print("\n5. RESULTS VALIDATION")
        print("-" * 40)
        
        validation_results = self._validate_analysis_results(analysis_results, ground_truth)
        
        # Step 6: Generate Visualizations
        print("\n6. VISUALIZATION GENERATION")
        print("-" * 40)
        
        visualizations = self._generate_visualizations(validation_results, analysis_results)
        
        # Step 7: Create Comprehensive Report
        print("\n7. REPORT GENERATION")
        print("-" * 40)
        
        final_report = self._generate_final_report(
            config, ground_truth, analysis_results, validation_results
        )
        
        # Step 8: Save Results
        print("\n8. SAVING RESULTS")
        print("-" * 40)
        
        self._save_all_results(
            config, ground_truth, simulation_results, 
            analysis_results, validation_results, final_report
        )
        
        print("\n" + "="*80)
        print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return {
            'config': config,
            'ground_truth': ground_truth,
            'simulation_results': simulation_results,
            'analysis_results': analysis_results,
            'validation_results': validation_results,
            'visualizations': visualizations,
            'final_report': final_report,
            'output_directory': str(self.output_dir.absolute())
        }
    
    def _setup_configuration(self, genome_length, read_length, coverage, 
                           efficiency_range, n_efficiency_points, gc_content,
                           methylation_rates, error_rate):
        """Setup and validate configuration parameters"""
        
        if methylation_rates is None:
            methylation_rates = {
                'cpg_methylation_rate': 0.75,
                'chg_methylation_rate': 0.05,
                'chh_methylation_rate': 0.02
            }
        
        # Create efficiency test points
        min_eff, max_eff = efficiency_range
        efficiency_points = self.validator.create_ground_truth_benchmarks(
            efficiency_range, n_efficiency_points
        )
        
        config = {
            'genome_parameters': {
                'genome_length': genome_length,
                'gc_content': gc_content,
                'methylation_rates': methylation_rates
            },
            'sequencing_parameters': {
                'read_length': read_length,
                'coverage': coverage,
                'error_rate': error_rate
            },
            'efficiency_parameters': {
                'efficiency_range': efficiency_range,
                'efficiency_points': efficiency_points,
                'n_points': len(efficiency_points)
            },
            'analysis_parameters': {
                'min_coverage_threshold': 5,
                'bootstrap_samples': 1000,
                'confidence_level': 0.95,
                'window_size': 500
            }
        }
        
        print(f"  ✓ Genome length: {genome_length:,} bp")
        print(f"  ✓ Read parameters: {read_length} bp reads, {coverage}x coverage")
        print(f"  ✓ Testing {len(efficiency_points)} conversion efficiencies: {min_eff:.1%} - {max_eff:.1%}")
        print(f"  ✓ GC content: {gc_content:.1%}")
        
        return config
    
    def _generate_ground_truth_data(self, config):
        """Generate ground truth reference genome and methylation profile"""
        
        # Import simulator class (in practice, would be from separate module)
        from bisulfite_simulation import BisulfiteSimulator
        
        # Initialize simulator
        genome_params = config['genome_parameters']
        seq_params = config['sequencing_parameters']
        
        self.simulator = BisulfiteSimulator(
            genome_length=genome_params['genome_length'],
            read_length=seq_params['read_length'],
            coverage=seq_params['coverage']
        )
        
        # Generate reference genome
        reference_genome = self.simulator.generate_reference_genome(
            gc_content=genome_params['gc_content']
        )
        
        # Generate methylation profile
        methylation_profile = self.simulator.generate_methylation_profile(
            **genome_params['methylation_rates']
        )
        
        # Calculate ground truth statistics
        stats = self._calculate_ground_truth_statistics(reference_genome, methylation_profile)
        
        ground_truth = {
            'reference_genome': reference_genome,
            'methylation_profile': methylation_profile,
            'statistics': stats,
            'simulator': self.simulator
        }
        
        print(f"  ✓ Generated {len(reference_genome):,} bp reference genome")
        print(f"  ✓ Created methylation profile for {len(methylation_profile):,} cytosines")
        print(f"  ✓ Ground truth statistics calculated")
        
        return ground_truth
    
    def _calculate_ground_truth_statistics(self, reference_genome, methylation_profile):
        """Calculate comprehensive statistics for ground truth data"""
        
        # Basic composition statistics
        composition = {
            'A': reference_genome.count('A'),
            'T': reference_genome.count('T'),
            'G': reference_genome.count('G'),
            'C': reference_genome.count('C')
        }
        
        total_bases = len(reference_genome)
        composition_pct = {base: count/total_bases for base, count in composition.items()}
        
        # Methylation statistics by context
        context_stats = {}
        for context in ['CpG', 'CHG', 'CHH']:
            context_positions = [pos for pos, data in methylation_profile.items() 
                               if data['context'] == context]
            methylated_positions = [pos for pos in context_positions 
                                  if methylation_profile[pos]['is_methylated']]
            
            context_stats[context] = {
                'total_sites': len(context_positions),
                'methylated_sites': len(methylated_positions),
                'methylation_rate': len(methylated_positions) / len(context_positions) if context_positions else 0
            }
        
        # CpG island analysis (simplified)
        cpg_density = self._calculate_cpg_density(reference_genome)
        
        return {
            'composition': composition,
            'composition_percentages': composition_pct,
            'context_statistics': context_stats,
            'cpg_density': cpg_density,
            'total_cytosines': len(methylation_profile),
            'genome_length': len(reference_genome)
        }
    
    def _calculate_cpg_density(self, sequence, window_size=100):
        """Calculate CpG density across the genome"""
        densities = []
        
        for i in range(0, len(sequence) - window_size, window_size):
            window = sequence[i:i+window_size]
            cpg_count = window.count('CG')
            c_count = window.count('C')
            g_count = window.count('G')
            
            # CpG density = observed CpG / expected CpG
            expected_cpg = (c_count * g_count) / len(window) if len(window) > 0 else 0
            density = cpg_count / expected_cpg if expected_cpg > 0 else 0
            densities.append(density)
        
        return {
            'mean_density': np.mean(densities),
            'std_density': np.std(densities),
            'densities': densities
        }
    
    def _simulate_bisulfite_data(self, ground_truth, config):
        """Simulate bisulfite sequencing data for different conversion efficiencies"""
        
        efficiency_points = config['efficiency_parameters']['efficiency_points']
        seq_params = config['sequencing_parameters']
        
        simulation_results = {}
        
        for i, efficiency in enumerate(efficiency_points):
            print(f"  Simulating efficiency {i+1}/{len(efficiency_points)}: {efficiency:.1%}")
            
            # Simulate bisulfite conversion
            converted_genome = self.simulator.simulate_bisulfite_conversion(
                conversion_efficiency=efficiency,
                incomplete_conversion_bias={'CpG': 1.0, 'CHG': 0.9, 'CHH': 0.8}
            )
            
            # Generate sequencing reads
            reads = self.simulator.generate_sequencing_reads(
                converted_genome, 
                error_rate=seq_params['error_rate']
            )
            
            # Store simulation data
            simulation_results[efficiency] = {
                'converted_genome': converted_genome,
                'reads': reads,
                'conversion_events': self.simulator.conversion_events.copy(),
                'true_efficiency': efficiency,
                'read_statistics': self._calculate_read_statistics(reads)
            }
        
        print(f"  ✓ Simulated {len(efficiency_points)} datasets")
        print(f"  ✓ Total reads generated: {sum(len(data['reads']) for data in simulation_results.values()):,}")
        
        return simulation_results
    
    def _calculate_read_statistics(self, reads):
        """Calculate statistics for generated reads"""
        
        if not reads:
            return {}
        
        lengths = [read['length'] for read in reads]
        start_positions = [read['start_pos'] for read in reads]
        
        return {
            'total_reads': len(reads),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'coverage_start': np.min(start_positions),
            'coverage_end': np.max([read['end_pos'] for read in reads]),
            'mean_start_pos': np.mean(start_positions)
        }
    
    def _analyze_conversion_efficiency(self, simulation_results, ground_truth):
        """Analyze conversion efficiency using multiple methods"""
        
        # Import analyzer class
        from bisulfite_metrics import ConversionEfficiencyAnalyzer, calculate_advanced_metrics
        
        self.analyzer = ConversionEfficiencyAnalyzer()
        
        reference_genome = ground_truth['reference_genome']
        methylation_profile = ground_truth['methylation_profile']
        
        analysis_results = {}
        
        for efficiency, sim_data in simulation_results.items():
            print(f"  Analyzing {efficiency:.1%} efficiency dataset...")
            
            reads = sim_data['reads']
            
            # Primary analysis
            primary_analysis = self.analyzer.analyze_reads(
                reads=reads,
                reference_genome=reference_genome,
                methylation_profile=methylation_profile,
                true_efficiency=efficiency
            )
            
            # Advanced metrics
            advanced_metrics = calculate_advanced_metrics(
                reads=reads,
                reference_genome=reference_genome,
                methylation_profile=methylation_profile,
                conversion_events=sim_data['conversion_events']
            )
            
            # Performance benchmarking
            performance_metrics = self._calculate_performance_metrics(
                primary_analysis, efficiency
            )
            
            # Combine all analysis results
            analysis_results[efficiency] = {
                'primary_analysis': primary_analysis,
                'advanced_metrics': advanced_metrics,
                'performance_metrics': performance_metrics,
                'dataset_info': {
                    'true_efficiency': efficiency,
                    'total_reads': len(reads),
                    'conversion_events': len(sim_data['conversion_events'])
                }
            }
        
        print(f"  ✓ Analyzed {len(analysis_results)} datasets")
        print(f"  ✓ Applied {len(['non_cpg', 'lambda_dna', 'chh', 'confidence_interval'])} efficiency measurement methods")
        
        return analysis_results
    
    def _calculate_performance_metrics(self, analysis, true_efficiency):
        """Calculate performance metrics for each analysis method"""
        
        methods = ['non_cpg_efficiency', 'lambda_efficiency', 'chh_efficiency']
        performance = {}
        
        for method in methods:
            method_data = analysis.get(method, {})
            estimated_eff = method_data.get('efficiency')
            
            if estimated_eff is not None:
                error = abs(estimated_eff - true_efficiency)
                relative_error = error / true_efficiency if true_efficiency > 0 else float('inf')
                bias = estimated_eff - true_efficiency
                
                performance[method] = {
                    'estimated_efficiency': estimated_eff,
                    'absolute_error': error,
                    'relative_error': relative_error,
                    'bias': bias,
                    'sample_size': method_data.get('total_sites', 0)
                }
        
        # Confidence interval performance
        ci_data = analysis.get('confidence_intervals', {})
        if 'mean_efficiency' in ci_data:
            estimated_eff = ci_data['mean_efficiency']
            ci_lower, ci_upper = ci_data.get('confidence_interval', (None, None))
            
            performance['confidence_interval'] = {
                'estimated_efficiency': estimated_eff,
                'absolute_error': abs(estimated_eff - true_efficiency),
                'relative_error': abs(estimated_eff - true_efficiency) / true_efficiency,
                'bias': estimated_eff - true_efficiency,
                'contains_true_value': ci_lower <= true_efficiency <= ci_upper if ci_lower and ci_upper else False,
                'interval_width': ci_upper - ci_lower if ci_lower and ci_upper else None
            }
        
        return performance
    
    def _validate_analysis_results(self, analysis_results, ground_truth):
        """Comprehensive validation of analysis results"""
        
        # Extract data for validation
        true_efficiencies = list(analysis_results.keys())
        
        # Restructure analysis results for validation framework
        validation_input = {}
        for efficiency, results in analysis_results.items():
            primary = results['primary_analysis']
            
            validation_input[efficiency] = {
                'non_cpg_efficiency': primary.get('non_cpg_efficiency', {}),
                'lambda_efficiency': primary.get('lambda_efficiency', {}),
                'chh_efficiency': primary.get('chh_efficiency', {}),
                'confidence_intervals': primary.get('confidence_intervals', {}),
                'summary_metrics': primary.get('summary_metrics', {})
            }
        
        # Run validation
        validation_results = self.validator.validate_method_accuracy(
            validation_input, true_efficiencies
        )
        
        # Add custom validation metrics
        custom_validation = self._perform_custom_validation(analysis_results, ground_truth)
        validation_results['custom_metrics'] = custom_validation
        
        print(f"  ✓ Validated {len(true_efficiencies)} efficiency measurements")
        print(f"  ✓ Analyzed {len([k for k in validation_results.keys() if k != 'consistency_analysis'])} methods")
        
        return validation_results
    
    def _perform_custom_validation(self, analysis_results, ground_truth):
        """Perform custom validation specific to bisulfite analysis"""
        
        # 1. Context-specific accuracy analysis
        context_accuracy = self._validate_context_specific_accuracy(analysis_results)
        
        # 2. Coverage-dependent performance
        coverage_performance = self._validate_coverage_dependence(analysis_results)
        
        # 3. Efficiency range sensitivity
        range_sensitivity = self._validate_efficiency_range_sensitivity(analysis_results)
        
        # 4. Method stability analysis
        stability_analysis = self._validate_method_stability(analysis_results)
        
        return {
            'context_accuracy': context_accuracy,
            'coverage_performance': coverage_performance,
            'range_sensitivity': range_sensitivity,
            'stability_analysis': stability_analysis
        }
    
    def _validate_context_specific_accuracy(self, analysis_results):
        """Validate accuracy for different cytosine contexts"""
        
        context_performance = {'CpG': [], 'CHG': [], 'CHH': []}
        
        for efficiency, results in analysis_results.items():
            context_analysis = results['primary_analysis'].get('context_analysis', {})
            
            for context, data in context_analysis.items():
                if context in context_performance and 'efficiency' in data:
                    estimated = data['efficiency']
                    error = abs(estimated - efficiency)
                    context_performance[context].append(error)
        
        # Calculate statistics for each context
        context_stats = {}
        for context, errors in context_performance.items():
            if errors:
                context_stats[context] = {
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'max_error': np.max(errors),
                    'n_samples': len(errors)
                }
        
        return context_stats
    
    def _validate_coverage_dependence(self, analysis_results):
        """Analyze how method performance depends on sequencing coverage"""
        
        coverage_data = []
        
        for efficiency, results in analysis_results.items():
            dataset_info = results['dataset_info']
            performance = results['performance_metrics']
            
            total_reads = dataset_info['total_reads']
            
            for method, perf_data in performance.items():
                if 'absolute_error' in perf_data:
                    coverage_data.append({
                        'method': method,
                        'coverage': total_reads,
                        'error': perf_data['absolute_error'],
                        'efficiency': efficiency
                    })
        
        # Group by method and calculate coverage-error correlation
        coverage_analysis = {}
        methods = set(row['method'] for row in coverage_data)
        
        for method in methods:
            method_data = [row for row in coverage_data if row['method'] == method]
            if len(method_data) >= 3:
                coverages = [row['coverage'] for row in method_data]
                errors = [row['error'] for row in method_data]
                
                correlation = np.corrcoef(coverages, errors)[0,1] if len(coverages) > 1 else 0
                
                coverage_analysis[method] = {
                    'coverage_error_correlation': correlation,
                    'mean_error': np.mean(errors),
                    'n_points': len(method_data)
                }
        
        return coverage_analysis
    
    def _validate_efficiency_range_sensitivity(self, analysis_results):
        """Analyze method performance across different efficiency ranges"""
        
        efficiencies = sorted(analysis_results.keys())
        
        # Define efficiency ranges
        ranges = {
            'low': [e for e in efficiencies if e < 0.95],
            'medium': [e for e in efficiencies if 0.95 <= e < 0.99],
            'high': [e for e in efficiencies if e >= 0.99]
        }
        
        range_performance = {}
        
        for range_name, range_effs in ranges.items():
            if not range_effs:
                continue
                
            range_errors = {}
            
            for efficiency in range_effs:
                performance = analysis_results[efficiency]['performance_metrics']
                
                for method, perf_data in performance.items():
                    if method not in range_errors:
                        range_errors[method] = []
                    
                    if 'absolute_error' in perf_data:
                        range_errors[method].append(perf_data['absolute_error'])
            
            # Calculate statistics for each method in this range
            range_stats = {}
            for method, errors in range_errors.items():
                if errors:
                    range_stats[method] = {
                        'mean_error': np.mean(errors),
                        'std_error': np.std(errors),
                        'max_error': np.max(errors),
                        'n_samples': len(errors)
                    }
            
            range_performance[range_name] = range_stats
        
        return range_performance
    
    def _validate_method_stability(self, analysis_results):
        """Analyze stability and consistency of methods across conditions"""
        
        # Calculate coefficient of variation for each method across all efficiencies
        method_stability = {}
        
        methods = ['non_cpg_efficiency', 'lambda_efficiency', 'chh_efficiency', 'confidence_interval']
        
        for method in methods:
            errors = []
            biases = []
            
            for efficiency, results in analysis_results.items():
                perf_data = results['performance_metrics'].get(method, {})
                
                if 'absolute_error' in perf_data:
                    errors.append(perf_data['absolute_error'])
                if 'bias' in perf_data:
                    biases.append(perf_data['bias'])
            
            if errors:
                method_stability[method] = {
                    'error_cv': np.std(errors) / np.mean(errors) if np.mean(errors) > 0 else float('inf'),
                    'mean_error': np.mean(errors),
                    'error_range': np.max(errors) - np.min(errors),
                    'bias_consistency': np.std(biases) if biases else 0,
                    'n_measurements': len(errors)
                }
        
        return method_stability
    
    def _generate_visualizations(self, validation_results, analysis_results):
        """Generate comprehensive visualizations"""
        
        print("  Creating validation plots...")
        figures = self.visualizer.create_accuracy_plots(
            validation_results, 
            save_path=str(self.output_dir / "validation_plots")
        )
        
        print("  Creating analysis summary plots...")
        summary_figures = self._create_analysis_summary_plots(analysis_results)
        
        print("  Creating method comparison plots...")
        comparison_figures = self._create_method_comparison_plots(analysis_results, validation_results)
        
        all_figures = {
            'validation_plots': figures,
            'summary_plots': summary_figures,
            'comparison_plots': comparison_figures
        }
        
        print(f"  ✓ Generated {sum(len(figs) for figs in all_figures.values())} visualization panels")
        
        return all_figures
    
    def _create_analysis_summary_plots(self, analysis_results):
        """Create summary plots showing analysis results"""
        
        # Extract data for plotting
        efficiencies = sorted(analysis_results.keys())
        
        # Plot 1: Efficiency estimation accuracy
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        methods = ['non_cpg_efficiency', 'lambda_efficiency', 'chh_efficiency']
        method_labels = ['Non-CpG', 'Lambda DNA', 'CHH Context']
        colors = ['blue', 'red', 'green']
        
        for method, label, color in zip(methods, method_labels, colors):
            estimated_effs = []
            for eff in efficiencies:
                est = analysis_results[eff]['primary_analysis'].get(method, {}).get('efficiency')
                estimated_effs.append(est if est is not None else np.nan)
            
            ax1.plot(efficiencies, estimated_effs, 'o-', label=label, color=color, 
                    linewidth=2, markersize=6)
        
        # Perfect prediction line
        ax1.plot(efficiencies, efficiencies, 'k--', alpha=0.5, label='Perfect Prediction')
        
        ax1.set_xlabel('True Conversion Efficiency')
        ax1.set_ylabel('Estimated Conversion Efficiency')
        ax1.set_title('Conversion Efficiency Estimation Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Method error comparison
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        method_errors = {method: [] for method in methods}
        
        for eff in efficiencies:
            for method in methods:
                perf_data = analysis_results[eff]['performance_metrics'].get(method, {})
                error = perf_data.get('absolute_error')
                method_errors[method].append(error if error is not None else np.nan)
        
        x_pos = np.arange(len(efficiencies))
        width = 0.25
        
        for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
            offset = (i - 1) * width
            ax2.bar(x_pos + offset, method_errors[method], width, 
                   label=label, color=color, alpha=0.7)
        
        ax2.set_xlabel('True Conversion Efficiency')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Method Error Comparison Across Efficiency Range')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{eff:.1%}' for eff in efficiencies], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        return {'efficiency_accuracy': fig1, 'error_comparison': fig2}
    
    def _create_method_comparison_plots(self, analysis_results, validation_results):
        """Create detailed method comparison visualizations"""
        
        # Plot 1: Performance metrics radar chart
        fig1, ax1 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        methods = [k for k in validation_results.keys() if k != 'consistency_analysis']
        metrics = ['r2_score', 'mae', 'mape', 'within_2pct_accuracy']
        metric_labels = ['R² Score', 'MAE', 'MAPE (%)', '±2% Accuracy (%)']
        
        # Normalize metrics for radar plot
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            if method in validation_results:
                data = validation_results[method]
                values = [
                    data.get('r2_score', 0),
                    1 - data.get('mae', 1),  # Invert MAE (higher is better)
                    1 - data.get('mape', 100) / 100,  # Invert MAPE
                    data.get('within_2pct_accuracy', 0) / 100
                ]
                values += values[:1]  # Complete the circle
                
                ax1.plot(angles, values, 'o-', linewidth=2, 
                        label=method.replace('_', ' ').title(), color=colors[i])
                ax1.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metric_labels)
        ax1.set_ylim(0, 1)
        ax1.set_title('Method Performance Comparison\n(Radar Chart)', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        return {'performance_radar': fig1}
    
    def _generate_final_report(self, config, ground_truth, analysis_results, validation_results):
        """Generate comprehensive final report"""
        
        print("  Creating comprehensive analysis report...")
        
        # Basic validation report
        basic_report = self.visualizer.create_comprehensive_report(validation_results)
        
        # Extended analysis report
        extended_report = self._create_extended_report(config, ground_truth, analysis_results, validation_results)
        
        # Combine reports
        final_report = f"""
{basic_report}

{extended_report}
        """
        
        return final_report.strip()
    
    def _create_extended_report(self, config, ground_truth, analysis_results, validation_results):
        """Create extended analysis report with additional insights"""
        
        report = []
        report.append("\n" + "="*80)
        report.append("EXTENDED ANALYSIS REPORT")
        report.append("="*80)
        
        # Configuration summary
        report.append("\nANALYSIS CONFIGURATION")
        report.append("-"*40)
        genome_params = config['genome_parameters']
        seq_params = config['sequencing_parameters']
        eff_params = config['efficiency_parameters']
        
        report.append(f"Genome length: {genome_params['genome_length']:,} bp")
        report.append(f"GC content: {genome_params['gc_content']:.1%}")
        report.append(f"Read length: {seq_params['read_length']} bp")
        report.append(f"Coverage: {seq_params['coverage']}x")
        report.append(f"Sequencing error rate: {seq_params['error_rate']:.1%}")
        report.append(f"Efficiency range tested: {eff_params['efficiency_range'][0]:.1%} - {eff_params['efficiency_range'][1]:.1%}")
        report.append(f"Number of test points: {eff_params['n_points']}")
        
        # Ground truth statistics
        report.append("\nGROUND TRUTH STATISTICS")
        report.append("-"*40)
        stats = ground_truth['statistics']
        
        report.append("Nucleotide composition:")
        for base, pct in stats['composition_percentages'].items():
            count = stats['composition'][base]
            report.append(f"  {base}: {count:,} ({pct:.1%})")
        
        report.append("\nMethylation by context:")
        for context, context_stats in stats['context_statistics'].items():
            total = context_stats['total_sites']
            methylated = context_stats['methylated_sites']
            rate = context_stats['methylation_rate']
            report.append(f"  {context}: {methylated:,}/{total:,} ({rate:.1%})")
        
        # Analysis performance summary
        report.append("\nANALYSIS PERFORMANCE SUMMARY")
        report.append("-"*40)
        
        # Best and worst performing conditions
        best_conditions = []
        worst_conditions = []
        
        for efficiency, results in analysis_results.items():
            consensus_error = 0
            n_methods = 0
            
            for method, perf_data in results['performance_metrics'].items():
                if 'absolute_error' in perf_data:
                    consensus_error += perf_data['absolute_error']
                    n_methods += 1
            
            if n_methods > 0:
                avg_error = consensus_error / n_methods
                best_conditions.append((efficiency, avg_error))
                worst_conditions.append((efficiency, avg_error))
        
        best_conditions.sort(key=lambda x: x[1])
        worst_conditions.sort(key=lambda x: x[1], reverse=True)
        
        if best_conditions:
            best_eff, best_err = best_conditions[0]
            report.append(f"Best performance: {best_eff:.1%} efficiency (avg error: {best_err:.6f})")
        
        if worst_conditions:
            worst_eff, worst_err = worst_conditions[0]
            report.append(f"Most challenging: {worst_eff:.1%} efficiency (avg error: {worst_err:.6f})")
        
        # Custom validation insights
        custom_metrics = validation_results.get('custom_metrics', {})
        
        if 'range_sensitivity' in custom_metrics:
            report.append("\nEFFICIENCY RANGE SENSITIVITY")
            report.append("-"*40)
            
            range_perf = custom_metrics['range_sensitivity']
            for range_name, methods in range_perf.items():
                report.append(f"{range_name.title()} efficiency range:")
                for method, stats in methods.items():
                    mean_err = stats['mean_error']
                    n_samples = stats['n_samples']
                    method_clean = method.replace('_', ' ').title()
                    report.append(f"  {method_clean}: {mean_err:.6f} avg error (n={n_samples})")
        
        # Recommendations and conclusions
        report.append("\nRECOMMENDATIONS AND CONCLUSIONS")
        report.append("-"*40)
        
        # Find most reliable method
        method_reliability = {}
        for method, validation_data in validation_results.items():
            if method != 'consistency_analysis' and 'r2_score' in validation_data:
                r2 = validation_data['r2_score']
                mae = validation_data['mae']
                within_2pct = validation_data.get('within_2pct_accuracy', 0)
                
                # Composite reliability score
                reliability = (r2 + (1 - mae) + within_2pct/100) / 3
                method_reliability[method] = reliability
        
        if method_reliability:
            best_method = max(method_reliability.items(), key=lambda x: x[1])
            report.append(f"Most reliable method: {best_method[0].replace('_', ' ').title()} "
                         f"(reliability score: {best_method[1]:.3f})")
        
        # Analysis limitations and considerations
        report.append("\nLIMITATIONS AND CONSIDERATIONS")
        report.append("-"*40)
        report.append("• Simulation assumes uniform error rates across genome")
        report.append("• Real bisulfite data may have additional biases not modeled")
        report.append("• Method performance may vary with different methylation patterns")
        report.append("• Coverage uniformity assumed - real data may have coverage bias")
        
