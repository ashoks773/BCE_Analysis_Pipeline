#!/usr/bin/env python3
"""
Bisulfite Conversion Efficiency Analysis Pipeline
Part 1: Simulated Data Generation

This module simulates bisulfite sequencing data with varying conversion efficiencies
to create ground truth datasets for validation.

Key Concepts:
- Bisulfite treatment converts unmethylated cytosines (C) to uracil (U), which reads as thymine (T)
- Methylated cytosines remain as cytosines
- Incomplete conversion leaves some unmethylated C's unconverted
- We simulate CpG and non-CpG contexts separately due to different methylation patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class BisulfiteSimulator:
    """
    Comprehensive bisulfite sequencing data simulator with realistic biological parameters
    """
    
    def __init__(self, genome_length=10000, read_length=150, coverage=30):
        """
        Initialize the simulator with basic parameters
        
        Args:
            genome_length (int): Length of reference genome to simulate
            read_length (int): Length of sequencing reads
            coverage (int): Average sequencing coverage depth
        """
        self.genome_length = genome_length
        self.read_length = read_length
        self.coverage = coverage
        self.reference_genome = None
        self.methylation_profile = None
        self.conversion_rates = {}
        
    def generate_reference_genome(self, gc_content=0.42):
        """
        Generate a realistic reference genome with specified GC content
        
        Args:
            gc_content (float): GC content of the genome (human genome ~42%)
        
        Returns:
            str: Reference genome sequence
        """
        print(f"Generating reference genome of length {self.genome_length} with GC content {gc_content:.2%}")
        
        # Generate nucleotides with realistic distribution
        nucleotides = ['A', 'T', 'G', 'C']
        # Adjust probabilities to achieve target GC content
        at_prob = (1 - gc_content) / 2
        gc_prob = gc_content / 2
        probs = [at_prob, at_prob, gc_prob, gc_prob]
        
        self.reference_genome = ''.join(np.random.choice(nucleotides, size=self.genome_length, p=probs))
        
        # Calculate actual GC content
        actual_gc = (self.reference_genome.count('G') + self.reference_genome.count('C')) / len(self.reference_genome)
        print(f"Actual GC content: {actual_gc:.2%}")
        
        return self.reference_genome
    
    def generate_methylation_profile(self, cpg_methylation_rate=0.75, chg_methylation_rate=0.05, chh_methylation_rate=0.02):
        """
        Generate realistic methylation profile based on biological patterns
        
        Args:
            cpg_methylation_rate (float): Methylation rate for CpG contexts
            chg_methylation_rate (float): Methylation rate for CHG contexts  
            chh_methylation_rate (float): Methylation rate for CHH contexts
        
        Returns:
            dict: Methylation status for each cytosine position
        """
        if self.reference_genome is None:
            raise ValueError("Reference genome must be generated first")
            
        print("Generating methylation profile...")
        self.methylation_profile = {}
        
        # Find all cytosine positions and their contexts
        for i, base in enumerate(self.reference_genome):
            if base == 'C':
                # Determine context
                context = self._get_cytosine_context(i)
                
                # Assign methylation status based on context
                if context == 'CpG':
                    is_methylated = np.random.random() < cpg_methylation_rate
                elif context == 'CHG':
                    is_methylated = np.random.random() < chg_methylation_rate
                elif context == 'CHH':
                    is_methylated = np.random.random() < chh_methylation_rate
                else:
                    is_methylated = False
                
                self.methylation_profile[i] = {
                    'context': context,
                    'is_methylated': is_methylated
                }
        
        # Print methylation statistics
        contexts = [data['context'] for data in self.methylation_profile.values()]
        methylated = [data['is_methylated'] for data in self.methylation_profile.values()]
        
        context_counts = Counter(contexts)
        methylated_by_context = defaultdict(int)
        total_by_context = defaultdict(int)
        
        for context, is_meth in zip(contexts, methylated):
            total_by_context[context] += 1
            if is_meth:
                methylated_by_context[context] += 1
        
        print(f"Total cytosines: {len(self.methylation_profile)}")
        for context in ['CpG', 'CHG', 'CHH']:
            if total_by_context[context] > 0:
                rate = methylated_by_context[context] / total_by_context[context]
                print(f"{context}: {total_by_context[context]} sites, {rate:.2%} methylated")
        
        return self.methylation_profile
    
    def _get_cytosine_context(self, position):
        """
        Determine the sequence context of a cytosine (CpG, CHG, CHH)
        
        Args:
            position (int): Position of cytosine in genome
            
        Returns:
            str: Context type ('CpG', 'CHG', 'CHH', or 'other')
        """
        if position + 2 >= len(self.reference_genome):
            return 'other'
            
        # Get the trinucleotide context
        trinuc = self.reference_genome[position:position+3]
        
        if len(trinuc) < 3:
            return 'other'
            
        if trinuc.startswith('CG'):
            return 'CpG'
        elif trinuc[1] == 'G' and trinuc[2] in 'ATCG':  # CHG
            return 'CHG'
        elif trinuc[1] in 'ATCG' and trinuc[2] in 'ATCG':  # CHH
            return 'CHH'
        else:
            return 'other'
    
    def simulate_bisulfite_conversion(self, conversion_efficiency=0.99, incomplete_conversion_bias=None):
        """
        Simulate bisulfite conversion with specified efficiency
        
        Args:
            conversion_efficiency (float): Overall conversion efficiency (0-1)
            incomplete_conversion_bias (dict): Context-specific conversion biases
            
        Returns:
            str: Bisulfite-converted genome sequence
        """
        if self.reference_genome is None or self.methylation_profile is None:
            raise ValueError("Reference genome and methylation profile must be generated first")
        
        print(f"Simulating bisulfite conversion with {conversion_efficiency:.1%} efficiency")
        
        # Default context-specific biases if not provided
        if incomplete_conversion_bias is None:
            incomplete_conversion_bias = {
                'CpG': 1.0,     # CpG sites convert at base rate
                'CHG': 0.9,     # CHG sites slightly harder to convert
                'CHH': 0.8,     # CHH sites hardest to convert
                'other': 0.85   # Other contexts intermediate
            }
        
        converted_genome = list(self.reference_genome)
        conversion_events = []
        
        for pos, cytosine_data in self.methylation_profile.items():
            context = cytosine_data['context']
            is_methylated = cytosine_data['is_methylated']
            
            # Methylated cytosines should remain as C
            if is_methylated:
                conversion_events.append({
                    'position': pos,
                    'original': 'C',
                    'converted': 'C',
                    'context': context,
                    'methylated': True,
                    'should_convert': False,
                    'did_convert': False
                })
                continue
            
            # Unmethylated cytosines should convert to T
            # Apply context-specific conversion efficiency
            context_efficiency = conversion_efficiency * incomplete_conversion_bias.get(context, 1.0)
            
            if np.random.random() < context_efficiency:
                # Successful conversion C -> T
                converted_genome[pos] = 'T'
                did_convert = True
            else:
                # Failed conversion, remains C
                did_convert = False
            
            conversion_events.append({
                'position': pos,
                'original': 'C',
                'converted': converted_genome[pos],
                'context': context,
                'methylated': False,
                'should_convert': True,
                'did_convert': did_convert
            })
        
        self.conversion_events = conversion_events
        converted_sequence = ''.join(converted_genome)
        
        # Calculate actual conversion rates
        should_convert = [e for e in conversion_events if e['should_convert']]
        did_convert = [e for e in should_convert if e['did_convert']]
        actual_efficiency = len(did_convert) / len(should_convert) if should_convert else 0
        
        print(f"Actual conversion efficiency: {actual_efficiency:.1%}")
        
        return converted_sequence
    
    def generate_sequencing_reads(self, converted_genome, error_rate=0.001):
        """
        Generate realistic sequencing reads from bisulfite-converted genome
        
        Args:
            converted_genome (str): Bisulfite-converted genome sequence
            error_rate (float): Sequencing error rate
            
        Returns:
            list: List of sequencing reads with metadata
        """
        print(f"Generating sequencing reads with {self.coverage}x coverage...")
        
        num_reads = int((len(converted_genome) * self.coverage) / self.read_length)
        reads = []
        
        for read_id in range(num_reads):
            # Random start position ensuring read fits within genome
            start_pos = np.random.randint(0, max(1, len(converted_genome) - self.read_length + 1))
            end_pos = min(start_pos + self.read_length, len(converted_genome))
            
            # Extract read sequence
            read_seq = converted_genome[start_pos:end_pos]
            
            # Add sequencing errors
            read_seq = self._add_sequencing_errors(read_seq, error_rate)
            
            # Generate quality scores (simplified)
            quality = self._generate_quality_scores(len(read_seq))
            
            reads.append({
                'read_id': f'read_{read_id:06d}',
                'sequence': read_seq,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'length': len(read_seq),
                'quality': quality
            })
        
        print(f"Generated {len(reads)} reads")
        return reads
    
    def _add_sequencing_errors(self, sequence, error_rate):
        """Add realistic sequencing errors to a read"""
        if error_rate <= 0:
            return sequence
            
        seq_list = list(sequence)
        nucleotides = ['A', 'T', 'G', 'C']
        
        for i in range(len(seq_list)):
            if np.random.random() < error_rate:
                # Introduce error - choose different nucleotide
                current = seq_list[i]
                alternatives = [n for n in nucleotides if n != current]
                seq_list[i] = np.random.choice(alternatives)
        
        return ''.join(seq_list)
    
    def _generate_quality_scores(self, length):
        """Generate realistic quality scores (Phred scores)"""
        # Simulate quality score distribution (higher at read start, lower at end)
        base_quality = 35
        quality_scores = []
        
        for i in range(length):
            # Quality degrades toward end of read
            pos_factor = max(0, 1 - (i / length) * 0.3)
            quality = int(base_quality * pos_factor + np.random.normal(0, 3))
            quality = max(10, min(40, quality))  # Clamp between 10-40
            quality_scores.append(quality)
        
        return quality_scores

def run_simulation_example():
    """
    Example of running the bisulfite simulation pipeline
    """
    print("=== Bisulfite Conversion Efficiency Simulation ===\n")
    
    # Initialize simulator
    simulator = BisulfiteSimulator(genome_length=5000, read_length=100, coverage=20)
    
    # Step 1: Generate reference genome
    ref_genome = simulator.generate_reference_genome(gc_content=0.42)
    print(f"Reference genome preview: {ref_genome[:100]}...\n")
    
    # Step 2: Generate methylation profile
    methylation_profile = simulator.generate_methylation_profile(
        cpg_methylation_rate=0.75,
        chg_methylation_rate=0.05, 
        chh_methylation_rate=0.02
    )
    print()
    
    # Step 3: Simulate different conversion efficiencies
    conversion_efficiencies = [0.95, 0.98, 0.99, 0.995]
    simulation_results = {}
    
    for efficiency in conversion_efficiencies:
        print(f"--- Simulating {efficiency:.1%} conversion efficiency ---")
        
        # Simulate bisulfite conversion
        converted_genome = simulator.simulate_bisulfite_conversion(
            conversion_efficiency=efficiency,
            incomplete_conversion_bias={'CpG': 1.0, 'CHG': 0.9, 'CHH': 0.8}
        )
        
        # Generate sequencing reads
        reads = simulator.generate_sequencing_reads(converted_genome, error_rate=0.001)
        
        # Store results
        simulation_results[efficiency] = {
            'converted_genome': converted_genome,
            'reads': reads,
            'conversion_events': simulator.conversion_events.copy()
        }
        print()
    
    return simulator, simulation_results

# Run example simulation
if __name__ == "__main__":
    simulator, results = run_simulation_example()
    print("Simulation complete! Results stored for analysis.")
    
    # Quick statistics
    print("\n=== Simulation Summary ===")
    for efficiency, data in results.items():
        num_reads = len(data['reads'])
        avg_read_length = np.mean([r['length'] for r in data['reads']])
        print(f"Efficiency {efficiency:.1%}: {num_reads} reads, avg length {avg_read_length:.1f}")
