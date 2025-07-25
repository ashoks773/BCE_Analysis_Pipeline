#!/usr/bin/env python3
"""
Bisulfite Conversion Efficiency Analysis Pipeline
Part 2: Metrics and Computational Approaches

This module implements various metrics and computational approaches to measure
bisulfite conversion efficiency from sequencing data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict, Counter
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')

class ConversionEfficiencyAnalyzer:
    """
    Comprehensive analyzer for measuring bisulfite conversion efficiency
    """
    
    def __init__(self):
        self.results = {}
        self.metrics = {}
        
    def analyze_reads(self, reads, reference_genome, methylation_profile, true_efficiency=None):
        """
        Comprehensive analysis of bisulfite sequencing reads to measure conversion efficiency
        
        Args:
            reads (list): List of sequencing reads
            reference_genome (str): Original reference genome
            methylation_profile (dict): True methylation status
            true_efficiency (float): Known conversion efficiency for validation
            
        Returns:
            dict: Comprehensive analysis results
        """
        print("Analyzing reads for conversion efficiency...")
        
        # Method 1: Non-CpG cytosine conversion rate
        non_cpg_efficiency = self._calculate_non_cpg_efficiency(reads, reference_genome, methylation_profile)
        
        # Method 2: Lambda DNA spike-in simulation (unmethylated control)
        lambda_efficiency = self._simulate_lambda_dna_efficiency(reads, reference_genome)
        
        # Method 3: CHH context analysis (should be mostly unmethylated)
        chh_efficiency = self._calculate_chh_efficiency(reads, reference_genome, methylation_profile)
        
        # Method 4: Position-specific conversion rates
        position_rates = self._calculate_position_specific_rates(reads, reference_genome, methylation_profile)
        
        # Method 5: Context-dependent conversion analysis
        context_analysis = self._analyze_context_dependent_conversion(reads, reference_genome, methylation_profile)
        
        # Method 6: Statistical confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(reads, reference_genome, methylation_profile)
        
        # Compile results
        analysis_results = {
            'non_cpg_efficiency': non_cpg_efficiency,
            'lambda_efficiency': lambda_efficiency,
            'chh_efficiency': chh_efficiency,
            'position_rates': position_rates,
            'context_analysis': context_analysis,
            'confidence_intervals': confidence_intervals,
            'true_efficiency': true_efficiency
        }
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(analysis_results)
        analysis_results['summary_metrics'] = summary_metrics
        
        return analysis_results
    
    def _calculate_non_cpg_efficiency(self, reads, reference_genome, methylation_profile):
        """
        Method 1: Calculate conversion efficiency using non-CpG cytosines
        
        Non-CpG cytosines should be mostly unmethylated in mammals and thus
        should convert to T after bisulfite treatment. This provides a direct
        measure of conversion efficiency.
        """
        print("  - Calculating non-CpG conversion efficiency...")
        
        non_cpg_positions = []
        for pos, data in methylation_profile.items():
            if data['context'] in ['CHG', 'CHH']:
                non_cpg_positions.append(pos)
        
        c_counts = 0  # Unconverted cytosines
        t_counts = 0  # Converted cytosines (now thymine)
        coverage_by_position = defaultdict(int)
        
        for read in reads:
            start_pos = read['start_pos']
            sequence = read['sequence']
            
            for i, base in enumerate(sequence):
                genome_pos = start_pos + i
                
                if genome_pos in non_cpg_positions:
                    coverage_by_position[genome_pos] += 1
                    
                    # Check if this position shows conversion
                    if reference_genome[genome_pos] == 'C':
                        if base == 'C':
                            c_counts += 1  # Unconverted
                        elif base == 'T':
                            t_counts += 1  # Converted
        
        total_counts = c_counts + t_counts
        efficiency = t_counts / total_counts if total_counts > 0 else 0
        
        # Calculate per-position conversion rates
        position_rates = {}
        for pos in non_cpg_positions:
            if coverage_by_position[pos] >= 5:  # Minimum coverage threshold
                pos_c = 0
                pos_t = 0
                
                for read in reads:
                    start_pos = read['start_pos']
                    end_pos = read['end_pos']
                    
                    if start_pos <= pos < end_pos:
                        read_pos = pos - start_pos
                        if read_pos < len(read['sequence']):
                            base = read['sequence'][read_pos]
                            if base == 'C':
                                pos_c += 1
                            elif base == 'T':
                                pos_t += 1
                
                if pos_c + pos_t > 0:
                    position_rates[pos] = pos_t / (pos_c + pos_t)
        
        return {
            'efficiency': efficiency,
            'c_counts': c_counts,
            't_counts': t_counts,
            'total_sites': len(non_cpg_positions),
            'covered_sites': len([p for p in non_cpg_positions if coverage_by_position[p] > 0]),
            'position_rates': position_rates,
            'average_coverage': np.mean(list(coverage_by_position.values())) if coverage_by_position else 0
        }
    
    def _simulate_lambda_dna_efficiency(self, reads, reference_genome):
        """
        Method 2: Simulate lambda DNA spike-in control
        
        Lambda DNA is unmethylated and serves as a positive control for conversion.
        We simulate this by treating all cytosines as unmethylated.
        """
        print("  - Simulating lambda DNA conversion efficiency...")
        
        c_counts = 0
        t_counts = 0
        total_c_positions = 0
        
        for read in reads:
            start_pos = read['start_pos']
            sequence = read['sequence']
            
            for i, base in enumerate(sequence):
                genome_pos = start_pos + i
                
                if genome_pos < len(reference_genome) and reference_genome[genome_pos] == 'C':
                    total_c_positions += 1
                    
                    if base == 'C':
                        c_counts += 1  # Unconverted
                    elif base == 'T':
                        t_counts += 1  # Converted
        
        efficiency = t_counts / (c_counts + t_counts) if (c_counts + t_counts) > 0 else 0
        
        return {
            'efficiency': efficiency,
            'c_counts': c_counts,
            't_counts': t_counts,
            'total_c_positions': total_c_positions
        }
    
    def _calculate_chh_efficiency(self, reads, reference_genome, methylation_profile):
        """
        Method 3: Calculate conversion efficiency using CHH context
        
        CHH cytosines are rarely methylated in mammals, making them good
        indicators of conversion efficiency.
        """
        print("  - Calculating CHH context conversion efficiency...")
        
        chh_positions = []
        for pos, data in methylation_profile.items():
            if data['context'] == 'CHH':
                chh_positions.append(pos)
        
        c_counts = 0
        t_counts = 0
        methylated_c = 0  # Should be very low
        
        for read in reads:
            start_pos = read['start_pos']
            sequence = read['sequence']
            
            for i, base in enumerate(sequence):
                genome_pos = start_pos + i
                
                if genome_pos in chh_positions:
                    if reference_genome[genome_pos] == 'C':
                        if base == 'C':
                            c_counts += 1
                            # Check if this was actually methylated
                            if methylation_profile[genome_pos]['is_methylated']:
                                methylated_c += 1
                        elif base == 'T':
                            t_counts += 1
        
        efficiency = t_counts / (c_counts + t_counts) if (c_counts + t_counts) > 0 else 0
        
        return {
            'efficiency': efficiency,
            'c_counts': c_counts,
            't_counts': t_counts,
            'methylated_c_counts': methylated_c,
            'total_chh_sites': len(chh_positions)
        }
    
    def _calculate_position_specific_rates(self, reads, reference_genome, methylation_profile):
        """
        Method 4: Calculate position-specific conversion rates across the genome
        """
        print("  - Calculating position-specific conversion rates...")
        
        position_data = defaultdict(lambda: {'C': 0, 'T': 0, 'other': 0, 'coverage': 0})
        
        for read in reads:
            start_pos = read['start_pos']
            sequence = read['sequence']
            
            for i, base in enumerate(sequence):
                genome_pos = start_pos + i
                
                if (genome_pos < len(reference_genome) and 
                    reference_genome[genome_pos] == 'C' and
                    genome_pos in methylation_profile):
                    
                    position_data[genome_pos]['coverage'] += 1
                    
                    if base == 'C':
                        position_data[genome_pos]['C'] += 1
                    elif base == 'T':
                        position_data[genome_pos]['T'] += 1
                    else:
                        position_data[genome_pos]['other'] += 1
        
        # Calculate conversion rates for positions with sufficient coverage
        conversion_rates = {}
        min_coverage = 5
        
        for pos, data in position_data.items():
            if data['coverage'] >= min_coverage:
                total_informative = data['C'] + data['T']
                if total_informative > 0:
                    conversion_rate = data['T'] / total_informative
                    conversion_rates[pos] = {
                        'rate': conversion_rate,
                        'coverage': data['coverage'],
                        'c_count': data['C'],
                        't_count': data['T'],
                        'context': methylation_profile[pos]['context'],
                        'methylated': methylation_profile[pos]['is_methylated']
                    }
        
        return conversion_rates
    
    def _analyze_context_dependent_conversion(self, reads, reference_genome, methylation_profile):
        """
        Method 5: Analyze conversion efficiency by sequence context
        """
        print("  - Analyzing context-dependent conversion...")
        
        context_data = defaultdict(lambda: {'C': 0, 'T': 0, 'positions': set()})
        
        for read in reads:
            start_pos = read['start_pos']
            sequence = read['sequence']
            
            for i, base in enumerate(sequence):
                genome_pos = start_pos + i
                
                if (genome_pos < len(reference_genome) and 
                    reference_genome[genome_pos] == 'C' and
                    genome_pos in methylation_profile):
                    
                    context = methylation_profile[genome_pos]['context']
                    is_methylated = methylation_profile[genome_pos]['is_methylated']
                    
                    # Only count unmethylated cytosines for conversion efficiency
                    if not is_methylated:
                        context_data[context]['positions'].add(genome_pos)
                        
                        if base == 'C':
                            context_data[context]['C'] += 1
                        elif base == 'T':
                            context_data[context]['T'] += 1
        
        # Calculate efficiency by context
        context_efficiencies = {}
        for context, data in context_data.items():
            total = data['C'] + data['T']
            if total > 0:
                efficiency = data['T'] / total
                context_efficiencies[context] = {
                    'efficiency': efficiency,
                    'c_counts': data['C'],
                    't_counts': data['T'],
                    'total_counts': total,
                    'unique_positions': len(data['positions'])
                }
        
        return context_efficiencies
    
    def _calculate_confidence_intervals(self, reads, reference_genome, methylation_profile, confidence=0.95):
        """
        Method 6: Calculate statistical confidence intervals for conversion efficiency
        """
        print("  - Calculating confidence intervals...")
        
        # Bootstrap sampling for confidence intervals
        n_bootstrap = 1000
        bootstrap_efficiencies = []
        
        # Get all conversion events (unmethylated C positions)
        conversion_events = []
        
        for read in reads:
            start_pos = read['start_pos']
            sequence = read['sequence']
            
            for i, base in enumerate(sequence):
                genome_pos = start_pos + i
                
                if (genome_pos < len(reference_genome) and 
                    reference_genome[genome_pos] == 'C' and
                    genome_pos in methylation_profile and
                    not methylation_profile[genome_pos]['is_methylated']):
                    
                    # Record conversion event
                    converted = 1 if base == 'T' else 0
                    conversion_events.append(converted)
        
        if len(conversion_events) < 10:
            return {'error': 'Insufficient data for confidence interval calculation'}
        
        # Bootstrap resampling
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(conversion_events, size=len(conversion_events), replace=True)
            bootstrap_efficiency = np.mean(bootstrap_sample)
            bootstrap_efficiencies.append(bootstrap_efficiency)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_efficiencies, lower_percentile)
        ci_upper = np.percentile(bootstrap_efficiencies, upper_percentile)
        mean_efficiency = np.mean(bootstrap_efficiencies)
        
        return {
            'mean_efficiency': mean_efficiency,
            'confidence_interval': (ci_lower, ci_upper),
            'confidence_level': confidence,
            'bootstrap_samples': len(bootstrap_efficiencies),
            'total_events': len(conversion_events)
        }
    
    def _calculate_summary_metrics(self, analysis_results):
        """
        Calculate summary metrics and performance statistics
        """
        print("  - Calculating summary metrics...")
        
        # Extract efficiency estimates from different methods
        methods_data = {
            'non_cpg': analysis_results.get('non_cpg_efficiency', {}).get('efficiency', None),
            'lambda_dna': analysis_results.get('lambda_efficiency', {}).get('efficiency', None),
            'chh_context': analysis_results.get('chh_efficiency', {}).get('efficiency', None),
            'confidence_interval': analysis_results.get('confidence_intervals', {}).get('mean_efficiency', None)
        }
        
        # Filter out None values
        valid_methods = {k: v for k, v in methods_data.items() if v is not None}
        
        if not valid_methods:
            return {'error': 'No valid efficiency estimates found'}
        
        # Calculate consensus efficiency
        efficiencies = list(valid_methods.values())
        consensus_efficiency = np.mean(efficiencies)
        efficiency_std = np.std(efficiencies)
        
        # Calculate method agreement (coefficient of variation)
        cv = efficiency_std / consensus_efficiency if consensus_efficiency > 0 else float('inf')
        
        # Performance metrics if true efficiency is known
        true_efficiency = analysis_results.get('true_efficiency')
        performance_metrics = {}
        
        if true_efficiency is not None:
            # Calculate accuracy metrics for each method
            method_errors = {}
            for method_name, estimated_eff in valid_methods.items():
                error = abs(estimated_eff - true_efficiency)
                relative_error = error / true_efficiency if true_efficiency > 0 else float('inf')
                method_errors[method_name] = {
                    'absolute_error': error,
                    'relative_error': relative_error,
                    'bias': estimated_eff - true_efficiency
                }
            
            # Overall performance
            consensus_error = abs(consensus_efficiency - true_efficiency)
            consensus_relative_error = consensus_error / true_efficiency if true_efficiency > 0 else float('inf')
            
            performance_metrics = {
                'method_errors': method_errors,
                'consensus_absolute_error': consensus_error,
                'consensus_relative_error': consensus_relative_error,
                'consensus_bias': consensus_efficiency - true_efficiency,
                'mae': consensus_error,  # Mean Absolute Error
                'rmse': np.sqrt(consensus_error**2)  # Root Mean Square Error
            }
        
        # Context-specific analysis summary
        context_summary = {}
        context_analysis = analysis_results.get('context_analysis', {})
        for context, data in context_analysis.items():
            context_summary[context] = {
                'efficiency': data.get('efficiency', 0),
                'sample_size': data.get('total_counts', 0),
                'unique_positions': data.get('unique_positions', 0)
            }
        
        return {
            'methods_efficiencies': valid_methods,
            'consensus_efficiency': consensus_efficiency,
            'efficiency_std': efficiency_std,
            'coefficient_variation': cv,
            'method_agreement': 'good' if cv < 0.05 else 'moderate' if cv < 0.1 else 'poor',
            'performance_metrics': performance_metrics,
            'context_summary': context_summary,
            'n_methods': len(valid_methods)
        }

def calculate_advanced_metrics(reads, reference_genome, methylation_profile, conversion_events=None):
    """
    Calculate advanced metrics for bisulfite conversion efficiency analysis
    
    Args:
        reads (list): Sequencing reads
        reference_genome (str): Reference genome sequence
        methylation_profile (dict): Methylation status data
        conversion_events (list): Optional conversion events data
        
    Returns:
        dict: Advanced metrics and statistics
    """
    print("Calculating advanced metrics...")
    
    metrics = {}
    
    # 1. Coverage distribution analysis
    coverage_stats = _analyze_coverage_distribution(reads, reference_genome, methylation_profile)
    metrics['coverage_analysis'] = coverage_stats
    
    # 2. Read length and quality analysis
    read_quality_stats = _analyze_read_quality(reads)
    metrics['read_quality'] = read_quality_stats
    
    # 3. Strand bias analysis
    strand_bias = _analyze_strand_bias(reads, reference_genome, methylation_profile)
    metrics['strand_bias'] = strand_bias
    
    # 4. Conversion rate variability
    conversion_variability = _analyze_conversion_variability(reads, reference_genome, methylation_profile)
    metrics['conversion_variability'] = conversion_variability
    
    # 5. Sequence composition effects
    composition_effects = _analyze_sequence_composition_effects(reads, reference_genome, methylation_profile)
    metrics['composition_effects'] = composition_effects
    
    return metrics

def _analyze_coverage_distribution(reads, reference_genome, methylation_profile):
    """Analyze coverage distribution across cytosine positions"""
    coverage_counts = defaultdict(int)
    
    for read in reads:
        start_pos = read['start_pos']
        end_pos = read['end_pos']
        
        for pos in range(start_pos, end_pos):
            if pos in methylation_profile:
                coverage_counts[pos] += 1
    
    coverages = list(coverage_counts.values())
    
    return {
        'mean_coverage': np.mean(coverages) if coverages else 0,
        'median_coverage': np.median(coverages) if coverages else 0,
        'coverage_std': np.std(coverages) if coverages else 0,
        'min_coverage': np.min(coverages) if coverages else 0,
        'max_coverage': np.max(coverages) if coverages else 0,
        'positions_covered': len(coverage_counts),
        'total_positions': len(methylation_profile)
    }

def _analyze_read_quality(reads):
    """Analyze read length and quality statistics"""
    lengths = [read['length'] for read in reads]
    qualities = []
    
    for read in reads:
        if 'quality' in read and read['quality']:
            qualities.extend(read['quality'])
    
    return {
        'mean_length': np.mean(lengths),
        'length_std': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'mean_quality': np.mean(qualities) if qualities else 0,
        'quality_std': np.std(qualities) if qualities else 0,
        'total_reads': len(reads)
    }

def _analyze_strand_bias(reads, reference_genome, methylation_profile):
    """Analyze potential strand bias in conversion"""
    # Simplified strand analysis - in real data, would need strand information
    # Here we simulate by position
    
    forward_conversions = 0
    reverse_conversions = 0
    forward_total = 0
    reverse_total = 0
    
    for read in reads:
        start_pos = read['start_pos']
        sequence = read['sequence']
        
        # Simulate strand assignment based on position (simplified)
        is_forward = start_pos % 2 == 0
        
        for i, base in enumerate(sequence):
            genome_pos = start_pos + i
            
            if (genome_pos < len(reference_genome) and 
                reference_genome[genome_pos] == 'C' and
                genome_pos in methylation_profile and
                not methylation_profile[genome_pos]['is_methylated']):
                
                if is_forward:
                    forward_total += 1
                    if base == 'T':
                        forward_conversions += 1
                else:
                    reverse_total += 1
                    if base == 'T':
                        reverse_conversions += 1
    
    forward_rate = forward_conversions / forward_total if forward_total > 0 else 0
    reverse_rate = reverse_conversions / reverse_total if reverse_total > 0 else 0
    
    return {
        'forward_rate': forward_rate,
        'reverse_rate': reverse_rate,
        'strand_difference': abs(forward_rate - reverse_rate),
        'forward_total': forward_total,
        'reverse_total': reverse_total
    }

def _analyze_conversion_variability(reads, reference_genome, methylation_profile):
    """Analyze variability in conversion rates across the genome"""
    
    # Calculate conversion rates in genomic windows
    window_size = 500
    window_rates = []
    
    for window_start in range(0, len(reference_genome) - window_size, window_size):
        window_end = window_start + window_size
        
        window_c = 0
        window_t = 0
        
        for read in reads:
            start_pos = read['start_pos']
            sequence = read['sequence']
            
            for i, base in enumerate(sequence):
                genome_pos = start_pos + i
                
                if (window_start <= genome_pos < window_end and
                    genome_pos < len(reference_genome) and 
                    reference_genome[genome_pos] == 'C' and
                    genome_pos in methylation_profile and
                    not methylation_profile[genome_pos]['is_methylated']):
                    
                    if base == 'C':
                        window_c += 1
                    elif base == 'T':
                        window_t += 1
        
        if window_c + window_t >= 10:  # Minimum counts for reliable estimate
            window_rate = window_t / (window_c + window_t)
            window_rates.append(window_rate)
    
    if len(window_rates) < 2:
        return {'error': 'Insufficient windows for variability analysis'}
    
    return {
        'mean_rate': np.mean(window_rates),
        'rate_std': np.std(window_rates),
        'rate_variance': np.var(window_rates),
        'min_rate': np.min(window_rates),
        'max_rate': np.max(window_rates),
        'rate_range': np.max(window_rates) - np.min(window_rates),
        'n_windows': len(window_rates)
    }

def _analyze_sequence_composition_effects(reads, reference_genome, methylation_profile):
    """Analyze how local sequence composition affects conversion"""
    
    composition_data = defaultdict(lambda: {'C': 0, 'T': 0})
    
    for read in reads:
        start_pos = read['start_pos']
        sequence = read['sequence']
        
        for i, base in enumerate(sequence):
            genome_pos = start_pos + i
            
            if (genome_pos < len(reference_genome) and 
                reference_genome[genome_pos] == 'C' and
                genome_pos in methylation_profile and
                not methylation_profile[genome_pos]['is_methylated']):
                
                # Get local sequence context (5 bp window)
                context_start = max(0, genome_pos - 2)
                context_end = min(len(reference_genome), genome_pos + 3)
                local_context = reference_genome[context_start:context_end]
                
                # Calculate GC content of local context
                gc_content = (local_context.count('G') + local_context.count('C')) / len(local_context)
                gc_bin = int(gc_content * 10) / 10  # Round to nearest 0.1
                
                if base == 'C':
                    composition_data[gc_bin]['C'] += 1
                elif base == 'T':
                    composition_data[gc_bin]['T'] += 1
    
    # Calculate conversion rates by GC content
    gc_conversion_rates = {}
    for gc_bin, counts in composition_data.items():
        total = counts['C'] + counts['T']
        if total >= 10:  # Minimum sample size
            rate = counts['T'] / total
            gc_conversion_rates[gc_bin] = {
                'rate': rate,
                'total_counts': total,
                'c_counts': counts['C'],
                't_counts': counts['T']
            }
    
    return gc_conversion_rates

# Example usage and testing functions
def run_comprehensive_analysis(simulator_results, true_efficiencies):
    """
    Run comprehensive analysis on simulated data
    
    Args:
        simulator_results (dict): Results from bisulfite simulator
        true_efficiencies (list): Known conversion efficiencies
        
    Returns:
        dict: Complete analysis results
    """
    analyzer = ConversionEfficiencyAnalyzer()
    all_results = {}
    
    for efficiency, sim_data in simulator_results.items():
        print(f"\n=== Analyzing {efficiency:.1%} efficiency data ===")
        
        # Get simulation components
        reads = sim_data['reads']
        converted_genome = sim_data['converted_genome']
        conversion_events = sim_data['conversion_events']
        
        # We need the original reference genome and methylation profile
        # These should be passed from the simulator
        reference_genome = None  # This should come from simulator
        methylation_profile = None  # This should come from simulator
        
        # Perform comprehensive analysis
        analysis_results = analyzer.analyze_reads(
            reads=reads,
            reference_genome=reference_genome,
            methylation_profile=methylation_profile,
            true_efficiency=efficiency
        )
        
        # Calculate advanced metrics
        advanced_metrics = calculate_advanced_metrics(
            reads=reads,
            reference_genome=reference_genome,
            methylation_profile=methylation_profile,
            conversion_events=conversion_events
        )
        
        # Combine results
        analysis_results['advanced_metrics'] = advanced_metrics
        all_results[efficiency] = analysis_results
    
    return all_results

if __name__ == "__main__":
    print("Bisulfite Conversion Efficiency Analysis Module")
    print("This module provides comprehensive metrics for measuring conversion efficiency.")
    print("Use in conjunction with the bisulfite simulator for complete analysis pipeline.")