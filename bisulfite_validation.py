#!/usr/bin/env python3
"""
Bisulfite Conversion Efficiency Analysis Pipeline
Part 3: Validation and Visualization

This module provides comprehensive validation methods and visualization tools
for assessing the accuracy of bisulfite conversion efficiency measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ValidationFramework:
    """
    Comprehensive validation framework for bisulfite conversion efficiency analysis
    """
    
    def __init__(self):
        self.validation_results = {}
        self.benchmark_data = {}
        
    def create_ground_truth_benchmarks(self, efficiency_range=(0.90, 0.999), n_points=10):
        """
        Create ground truth benchmarks with known conversion efficiencies
        
        Args:
            efficiency_range (tuple): Min and max conversion efficiencies
            n_points (int): Number of benchmark points to create
            
        Returns:
            list: List of benchmark conversion efficiencies
        """
        print("Creating ground truth benchmarks...")
        
        min_eff, max_eff = efficiency_range
        
        # Create both linear and non-linear spacing for comprehensive testing
        linear_points = np.linspace(min_eff, max_eff, n_points // 2)
        
        # Add some challenging points near high efficiency
        challenging_points = np.array([0.95, 0.98, 0.985, 0.99, 0.995, 0.999])
        challenging_points = challenging_points[challenging_points <= max_eff]
        
        # Combine and sort
        all_points = np.unique(np.concatenate([linear_points, challenging_points]))
        all_points = all_points[:n_points]  # Limit to requested number
        
        print(f"Created {len(all_points)} benchmark points: {all_points}")
        return all_points.tolist()
    
    def validate_method_accuracy(self, analysis_results, true_efficiencies):
        """
        Validate the accuracy of conversion efficiency measurement methods
        
        Args:
            analysis_results (dict): Results from efficiency analysis
            true_efficiencies (list): Known true conversion efficiencies
            
        Returns:
            dict: Comprehensive validation metrics
        """
        print("Validating method accuracy...")
        
        validation_metrics = {}
        
        # Extract estimated efficiencies from different methods
        methods = ['non_cpg', 'lambda_dna', 'chh_context', 'confidence_interval']
        
        for method in methods:
            estimated_effs = []
            true_effs = []
            
            for true_eff, results in zip(true_efficiencies, analysis_results.values()):
                if method == 'confidence_interval':
                    est_eff = results.get('confidence_intervals', {}).get('mean_efficiency')
                else:
                    method_key = f'{method}_efficiency' if method != 'lambda_dna' else 'lambda_efficiency'
                    est_eff = results.get(method_key, {}).get('efficiency')
                
                if est_eff is not None:
                    estimated_effs.append(est_eff)
                    true_effs.append(true_eff)
            
            if len(estimated_effs) >= 3:  # Minimum points for meaningful validation
                method_validation = self._calculate_validation_metrics(true_effs, estimated_effs)
                method_validation['method_name'] = method
                method_validation['n_points'] = len(estimated_effs)
                validation_metrics[method] = method_validation
        
        # Overall consensus validation
        consensus_effs = []
        consensus_true = []
        
        for true_eff, results in zip(true_efficiencies, analysis_results.values()):
            summary = results.get('summary_metrics', {})
            consensus_eff = summary.get('consensus_efficiency')
            
            if consensus_eff is not None:
                consensus_effs.append(consensus_eff)
                consensus_true.append(true_eff)
        
        if len(consensus_effs) >= 3:
            consensus_validation = self._calculate_validation_metrics(consensus_true, consensus_effs)
            consensus_validation['method_name'] = 'consensus'
            consensus_validation['n_points'] = len(consensus_effs)
            validation_metrics['consensus'] = consensus_validation
        
        # Cross-method consistency analysis
        consistency_analysis = self._analyze_method_consistency(analysis_results)
        validation_metrics['consistency_analysis'] = consistency_analysis
        
        return validation_metrics
    
    def _calculate_validation_metrics(self, true_values, predicted_values):
        """Calculate comprehensive validation metrics"""
        
        true_arr = np.array(true_values)
        pred_arr = np.array(predicted_values)
        
        # Basic error metrics
        mae = mean_absolute_error(true_arr, pred_arr)
        mse = mean_squared_error(true_arr, pred_arr)
        rmse = np.sqrt(mse)
        
        # Relative metrics
        mape = np.mean(np.abs((true_arr - pred_arr) / true_arr)) * 100  # Mean Absolute Percentage Error
        
        # Correlation metrics
        r2 = r2_score(true_arr, pred_arr)
        pearson_r, pearson_p = stats.pearsonr(true_arr, pred_arr)
        spearman_r, spearman_p = stats.spearmanr(true_arr, pred_arr)
        
        # Bias metrics
        bias = np.mean(pred_arr - true_arr)
        relative_bias = bias / np.mean(true_arr) * 100
        
        # Residual analysis
        residuals = pred_arr - true_arr
        residual_std = np.std(residuals)
        
        # Confidence metrics (within X% accuracy)
        within_1pct = np.mean(np.abs(residuals) <= 0.01) * 100
        within_2pct = np.mean(np.abs(residuals) <= 0.02) * 100
        within_5pct = np.mean(np.abs(residuals) <= 0.05) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2,
            'pearson_correlation': pearson_r,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_r,
            'spearman_p_value': spearman_p,
            'bias': bias,
            'relative_bias_percent': relative_bias,
            'residual_std': residual_std,
            'within_1pct_accuracy': within_1pct,
            'within_2pct_accuracy': within_2pct,
            'within_5pct_accuracy': within_5pct,
            'true_values': true_values,
            #'predicted_values': predicted_values.tolist(),
	    'predicted_values': list(predicted_values),
            'residuals': residuals.tolist()
        }
    
    def _analyze_method_consistency(self, analysis_results):
        """Analyze consistency between different measurement methods"""
        
        # Extract all method estimates for each sample
        method_estimates = defaultdict(list)
        sample_ids = list(analysis_results.keys())
        
        for sample_id, results in analysis_results.items():
            # Extract estimates from each method
            non_cpg = results.get('non_cpg_efficiency', {}).get('efficiency')
            lambda_dna = results.get('lambda_efficiency', {}).get('efficiency')
            chh = results.get('chh_efficiency', {}).get('efficiency')
            ci = results.get('confidence_intervals', {}).get('mean_efficiency')
            
            if non_cpg is not None:
                method_estimates['non_cpg'].append(non_cpg)
            if lambda_dna is not None:
                method_estimates['lambda_dna'].append(lambda_dna)
            if chh is not None:
                method_estimates['chh'].append(chh)
            if ci is not None:
                method_estimates['confidence_interval'].append(ci)
        
        # Calculate pairwise correlations between methods
        method_correlations = {}
        methods = list(method_estimates.keys())
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                if (len(method_estimates[method1]) >= 3 and 
                    len(method_estimates[method2]) >= 3):
                    
                    # Match up samples that have both estimates
                    paired_est1 = []
                    paired_est2 = []
                    
                    min_len = min(len(method_estimates[method1]), len(method_estimates[method2]))
                    paired_est1 = method_estimates[method1][:min_len]
                    paired_est2 = method_estimates[method2][:min_len]
                    
                    if len(paired_est1) >= 3:
                        corr, p_val = stats.pearsonr(paired_est1, paired_est2)
                        method_correlations[f'{method1}_vs_{method2}'] = {
                            'correlation': corr,
                            'p_value': p_val,
                            'n_samples': len(paired_est1)
                        }
        
        # Calculate coefficient of variation across methods for each sample
        sample_cvs = []
        for sample_id, results in analysis_results.items():
            estimates = []
            summary = results.get('summary_metrics', {})
            method_effs = summary.get('methods_efficiencies', {})
            
            estimates = list(method_effs.values())
            
            if len(estimates) >= 2:
                cv = np.std(estimates) / np.mean(estimates) if np.mean(estimates) > 0 else 0
                sample_cvs.append(cv)
        
        mean_cv = np.mean(sample_cvs) if sample_cvs else 0
        
        return {
            'method_correlations': method_correlations,
            'mean_coefficient_variation': mean_cv,
            'cv_distribution': sample_cvs,
            'n_samples_analyzed': len(sample_cvs)
        }

class VisualizationSuite:
    """
    Comprehensive visualization suite for bisulfite conversion efficiency analysis
    """
    
    def __init__(self, figsize=(12, 8), dpi=300):
        self.figsize = figsize
        self.dpi = dpi
        
    def create_accuracy_plots(self, validation_results, save_path=None):
        """
        Create comprehensive accuracy validation plots
        
        Args:
            validation_results (dict): Results from validation framework
            save_path (str): Optional path to save plots
            
        Returns:
            dict: Figure objects for further customization
        """
        print("Creating accuracy validation plots...")
        
        figures = {}
        
        # 1. Method comparison scatter plots
        fig1 = self._plot_method_accuracy_comparison(validation_results)
        figures['method_comparison'] = fig1
        
        # 2. Error distribution plots
        fig2 = self._plot_error_distributions(validation_results)
        figures['error_distributions'] = fig2
        
        # 3. Residual analysis plots
        fig3 = self._plot_residual_analysis(validation_results)
        figures['residual_analysis'] = fig3
        
        # 4. Method consistency plots
        fig4 = self._plot_method_consistency(validation_results)
        figures['method_consistency'] = fig4
        
        if save_path:
            for name, fig in figures.items():
                fig.savefig(f"{save_path}_{name}.png", dpi=self.dpi, bbox_inches='tight')
                print(f"Saved {name} plot to {save_path}_{name}.png")
        
        return figures
    
    def _plot_method_accuracy_comparison(self, validation_results):
        """Create scatter plots comparing predicted vs true values for each method"""
        
        methods_to_plot = [k for k in validation_results.keys() 
                          if k != 'consistency_analysis' and 'true_values' in validation_results[k]]
        
        n_methods = len(methods_to_plot)
        if n_methods == 0:
            print("No methods with sufficient data for plotting")
            return None
        
        # Create subplots
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), dpi=self.dpi)
        if n_methods == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if n_methods == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, method in enumerate(methods_to_plot):
            ax = axes[i] if n_methods > 1 else axes[0]
            
            data = validation_results[method]
            true_vals = np.array(data['true_values'])
            pred_vals = np.array(data['predicted_values'])
            
            # Scatter plot
            ax.scatter(true_vals, pred_vals, alpha=0.7, s=50)
            
            # Perfect prediction line
            min_val = min(np.min(true_vals), np.min(pred_vals))
            max_val = max(np.max(true_vals), np.max(pred_vals))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect prediction')
            
            # Fit line
            z = np.polyfit(true_vals, pred_vals, 1)
            p = np.poly1d(z)
            ax.plot(true_vals, p(true_vals), 'g-', alpha=0.8, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
            
            # Formatting
            ax.set_xlabel('True Conversion Efficiency')
            ax.set_ylabel('Predicted Conversion Efficiency')
            ax.set_title(f'{method.replace("_", " ").title()} Method\n'
                        f'R² = {data["r2_score"]:.3f}, RMSE = {data["rmse"]:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
        
        # Hide unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Method Accuracy Comparison: Predicted vs True Conversion Efficiency', 
                     fontsize=16, y=1.02)
        
        return fig
    
    def _plot_error_distributions(self, validation_results):
        """Create error distribution plots for each method"""
        
        methods_to_plot = [k for k in validation_results.keys() 
                          if k != 'consistency_analysis' and 'residuals' in validation_results[k]]
        
        if not methods_to_plot:
            print("No methods with residual data for plotting")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # 1. Residual distributions (histogram)
        all_residuals = []
        method_names = []
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods_to_plot)))
        
        for i, method in enumerate(methods_to_plot):
            residuals = validation_results[method]['residuals']
            all_residuals.extend(residuals)
            method_names.extend([method.replace('_', ' ').title()] * len(residuals))
            
            ax1.hist(residuals, bins=15, alpha=0.6, label=method.replace('_', ' ').title(), 
                    color=colors[i], density=True)
        
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Perfect accuracy')
        ax1.set_xlabel('Residuals (Predicted - True)')
        ax1.set_ylabel('Density')
        ax1.set_title('Error Distribution by Method')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot of absolute errors
        abs_errors_by_method = []
        method_labels = []
        
        for method in methods_to_plot:
            residuals = np.array(validation_results[method]['residuals'])
            abs_errors = np.abs(residuals)
            abs_errors_by_method.append(abs_errors)
            method_labels.append(method.replace('_', ' ').title())
        
        bp = ax2.boxplot(abs_errors_by_method, labels=method_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Absolute Error Distribution')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Accuracy within tolerance levels
        tolerance_levels = [0.01, 0.02, 0.05]
        method_accuracies = {method: [] for method in methods_to_plot}
        
        for method in methods_to_plot:
            data = validation_results[method]
            accuracies = [
                data.get('within_1pct_accuracy', 0),
                data.get('within_2pct_accuracy', 0),
                data.get('within_5pct_accuracy', 0)
            ]
            method_accuracies[method] = accuracies
        
        x = np.arange(len(tolerance_levels))
        width = 0.8 / len(methods_to_plot)
        
        for i, method in enumerate(methods_to_plot):
            offset = (i - len(methods_to_plot)/2 + 0.5) * width
            ax3.bar(x + offset, method_accuracies[method], width, 
                   label=method.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.7)
        
        ax3.set_xlabel('Tolerance Level')
        ax3.set_ylabel('Percentage of Predictions (%)')
        ax3.set_title('Accuracy Within Tolerance Levels')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['±1%', '±2%', '±5%'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance metrics comparison
        metrics = ['mae', 'rmse', 'mape', 'r2_score']
        metric_labels = ['MAE', 'RMSE', 'MAPE (%)', 'R²']
        
        # Normalize metrics for comparison (except R²)
        normalized_data = []
        for method in methods_to_plot:
            data = validation_results[method]
            method_metrics = [
                data.get('mae', 0),
                data.get('rmse', 0),
                data.get('mape', 0),
                data.get('r2_score', 0)
            ]
            normalized_data.append(method_metrics)
        
        # Create radar-like comparison
        x_pos = np.arange(len(metric_labels))
        for i, method in enumerate(methods_to_plot):
            ax4.plot(x_pos, normalized_data[i], 'o-', 
                    label=method.replace('_', ' ').title(),
                    color=colors[i], linewidth=2, markersize=6)
        
        ax4.set_xlabel('Performance Metrics')
        ax4.set_ylabel('Metric Value')
        ax4.set_title('Performance Metrics Comparison')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(metric_labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_residual_analysis(self, validation_results):
        """Create detailed residual analysis plots"""
        
        methods_to_plot = [k for k in validation_results.keys() 
                          if k != 'consistency_analysis' and 'residuals' in validation_results[k]]
        
        if not methods_to_plot:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods_to_plot)))
        
        # 1. Residuals vs Predicted Values
        for i, method in enumerate(methods_to_plot):
            data = validation_results[method]
            predicted = np.array(data['predicted_values'])
            residuals = np.array(data['residuals'])
            
            axes[0,0].scatter(predicted, residuals, alpha=0.6, 
                            label=method.replace('_', ' ').title(),
                            color=colors[i], s=30)
        
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
        axes[0,0].set_xlabel('Predicted Values')
        axes[0,0].set_ylabel('Residuals')
        axes[0,0].set_title('Residuals vs Predicted Values')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Q-Q plots for normality assessment
        from scipy.stats import probplot
        
        for i, method in enumerate(methods_to_plot):
            residuals = np.array(validation_results[method]['residuals'])
            probplot(residuals, dist="norm", plot=axes[0,1])
        
        axes[0,1].set_title('Q-Q Plot: Residual Normality')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Residuals vs True Values
        for i, method in enumerate(methods_to_plot):
            data = validation_results[method]
            true_vals = np.array(data['true_values'])
            residuals = np.array(data['residuals'])
            
            axes[1,0].scatter(true_vals, residuals, alpha=0.6,
                            label=method.replace('_', ' ').title(),
                            color=colors[i], s=30)
        
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
        axes[1,0].set_xlabel('True Values')
        axes[1,0].set_ylabel('Residuals')
        axes[1,0].set_title('Residuals vs True Values')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Absolute residuals vs predicted (heteroscedasticity check)
        for i, method in enumerate(methods_to_plot):
            data = validation_results[method]
            predicted = np.array(data['predicted_values'])
            abs_residuals = np.abs(np.array(data['residuals']))
            
            axes[1,1].scatter(predicted, abs_residuals, alpha=0.6,
                            label=method.replace('_', ' ').title(),
                            color=colors[i], s=30)
        
        axes[1,1].set_xlabel('Predicted Values')
        axes[1,1].set_ylabel('|Residuals|')
        axes[1,1].set_title('Absolute Residuals vs Predicted')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_method_consistency(self, validation_results):
        """Create method consistency analysis plots"""
        
        consistency_data = validation_results.get('consistency_analysis')
        if not consistency_data:
            print("No consistency analysis data available")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # 1. Method correlation heatmap
        correlations = consistency_data.get('method_correlations', {})
        if correlations:
            # Extract correlation matrix
            methods = set()
            for pair in correlations.keys():
                method1, method2 = pair.split('_vs_')
                methods.add(method1)
                methods.add(method2)
            
            methods = sorted(list(methods))
            n_methods = len(methods)
            
            if n_methods > 1:
                corr_matrix = np.eye(n_methods)
                
                for i, method1 in enumerate(methods):
                    for j, method2 in enumerate(methods):
                        if i != j:
                            pair_key1 = f"{method1}_vs_{method2}"
                            pair_key2 = f"{method2}_vs_{method1}"
                            
                            if pair_key1 in correlations:
                                corr_matrix[i,j] = correlations[pair_key1]['correlation']
                            elif pair_key2 in correlations:
                                corr_matrix[i,j] = correlations[pair_key2]['correlation']
                
                im = ax1.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax1.set_xticks(range(n_methods))
                ax1.set_yticks(range(n_methods))
                ax1.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
                ax1.set_yticklabels([m.replace('_', ' ').title() for m in methods])
                
                # Add correlation values as text
                for i in range(n_methods):
                    for j in range(n_methods):
                        text = ax1.text(j, i, f'{corr_matrix[i,j]:.2f}',
                                       ha="center", va="center", color="black", fontweight='bold')
                
                plt.colorbar(im, ax=ax1)
                ax1.set_title('Inter-Method Correlations')
        
        # 2. Coefficient of variation distribution
        cv_data = consistency_data.get('cv_distribution', [])
        if cv_data:
            ax2.hist(cv_data, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=np.mean(cv_data), color='red', linestyle='--', 
                       label=f'Mean CV = {np.mean(cv_data):.3f}')
            ax2.set_xlabel('Coefficient of Variation')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Method Agreement Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Pairwise method comparisons
        if len(correlations) > 0:
            pairs = list(correlations.keys())[:6]  # Show top 6 pairs
            corr_values = [correlations[pair]['correlation'] for pair in pairs]
            n_samples = [correlations[pair]['n_samples'] for pair in pairs]
            
            # Create bubble chart
            x_pos = range(len(pairs))
            colors_bubble = plt.cm.viridis(np.array(n_samples) / max(n_samples))
            
            scatter = ax3.scatter(x_pos, corr_values, s=[n*20 for n in n_samples], 
                                c=colors_bubble, alpha=0.6, edgecolors='black')
            
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([pair.replace('_vs_', ' vs\n').replace('_', ' ').title() 
                               for pair in pairs], rotation=45)
            ax3.set_ylabel('Correlation Coefficient')
            ax3.set_title('Pairwise Method Correlations\n(Bubble size = sample size)')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(-0.1, 1.1)
        
        # 4. Method reliability summary
        # Create a summary plot showing key metrics for each method
        method_names = []
        mae_values = []
        r2_values = []
        within_2pct = []
        
        for method, data in validation_results.items():
            if method != 'consistency_analysis' and 'mae' in data:
                method_names.append(method.replace('_', ' ').title())
                mae_values.append(data.get('mae', 0))
                r2_values.append(data.get('r2_score', 0))
                within_2pct.append(data.get('within_2pct_accuracy', 0))
        
        if method_names:
            x_pos = np.arange(len(method_names))
            
            # Create twin axis for different scales
            ax4_twin = ax4.twinx()
            
            # Plot MAE as bars
            bars1 = ax4.bar(x_pos - 0.2, mae_values, 0.4, label='MAE', alpha=0.7, color='lightcoral')
            # Plot R² as line
            line1 = ax4_twin.plot(x_pos, r2_values, 'go-', label='R²', linewidth=2, markersize=6)
            
            ax4.set_xlabel('Methods')
            ax4.set_ylabel('Mean Absolute Error', color='red')
            ax4_twin.set_ylabel('R² Score', color='green')
            ax4.set_title('Method Performance Summary')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(method_names, rotation=45)
            
            # Add accuracy percentages as text
            for i, (mae, r2, acc) in enumerate(zip(mae_values, r2_values, within_2pct)):
                ax4.text(i, mae + max(mae_values)*0.05, f'{acc:.0f}%', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_report(self, validation_results):
        """
        Create a comprehensive text report of validation results
        
        Args:
            validation_results (dict): Results from validation framework
            
        Returns:
            str: Formatted validation report
        """
        print("Generating comprehensive validation report...")
        
        report = []
        report.append("=" * 80)
        report.append("BISULFITE CONVERSION EFFICIENCY VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall summary
        methods_analyzed = [k for k in validation_results.keys() if k != 'consistency_analysis']
        report.append(f"Total methods analyzed: {len(methods_analyzed)}")
        report.append("")
        
        # Method-specific results
        for method, data in validation_results.items():
            if method == 'consistency_analysis':
                continue
                
            report.append(f"METHOD: {method.replace('_', ' ').upper()}")
            report.append("-" * 40)
            
            if 'n_points' in data:
                report.append(f"Sample size: {data['n_points']} data points")
            
            # Accuracy metrics
            report.append("Accuracy Metrics:")
            if 'mae' in data:
                report.append(f"  • Mean Absolute Error (MAE): {data['mae']:.6f}")
            if 'rmse' in data:
                report.append(f"  • Root Mean Square Error (RMSE): {data['rmse']:.6f}")
            if 'mape' in data:
                report.append(f"  • Mean Absolute Percentage Error (MAPE): {data['mape']:.2f}%")
            if 'r2_score' in data:
                report.append(f"  • R² Score: {data['r2_score']:.4f}")
            
            # Correlation metrics
            report.append("Correlation Metrics:")
            if 'pearson_correlation' in data:
                report.append(f"  • Pearson correlation: {data['pearson_correlation']:.4f} "
                            f"(p-value: {data.get('pearson_p_value', 'N/A'):.2e})")
            if 'spearman_correlation' in data:
                report.append(f"  • Spearman correlation: {data['spearman_correlation']:.4f}")
            
            # Bias analysis
            if 'bias' in data:
                report.append("Bias Analysis:")
                report.append(f"  • Absolute bias: {data['bias']:.6f}")
                report.append(f"  • Relative bias: {data.get('relative_bias_percent', 0):.2f}%")
            
            # Accuracy within tolerance
            report.append("Accuracy within tolerance:")
            if 'within_1pct_accuracy' in data:
                report.append(f"  • Within ±1%: {data['within_1pct_accuracy']:.1f}%")
            if 'within_2pct_accuracy' in data:
                report.append(f"  • Within ±2%: {data['within_2pct_accuracy']:.1f}%")
            if 'within_5pct_accuracy' in data:
                report.append(f"  • Within ±5%: {data['within_5pct_accuracy']:.1f}%")
            
            report.append("")
        
        # Consistency analysis
        consistency_data = validation_results.get('consistency_analysis')
        if consistency_data:
            report.append("INTER-METHOD CONSISTENCY ANALYSIS")
            report.append("-" * 40)
            
            mean_cv = consistency_data.get('mean_coefficient_variation', 0)
            report.append(f"Mean coefficient of variation across methods: {mean_cv:.4f}")
            
            if mean_cv < 0.05:
                consistency_rating = "Excellent"
            elif mean_cv < 0.1:
                consistency_rating = "Good"
            elif mean_cv < 0.2:
                consistency_rating = "Moderate"
            else:
                consistency_rating = "Poor"
            
            report.append(f"Method consistency rating: {consistency_rating}")
            
            correlations = consistency_data.get('method_correlations', {})
            if correlations:
                report.append("\nPairwise method correlations:")
                for pair, corr_data in correlations.items():
                    correlation = corr_data['correlation']
                    n_samples = corr_data['n_samples']
                    pair_clean = pair.replace('_vs_', ' vs ').replace('_', ' ').title()
                    report.append(f"  • {pair_clean}: r = {correlation:.3f} (n = {n_samples})")
            
            report.append("")
        
        # Overall recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        best_method = None
        best_r2 = -1
        
        for method, data in validation_results.items():
            if method != 'consistency_analysis' and 'r2_score' in data:
                if data['r2_score'] > best_r2:
                    best_r2 = data['r2_score']
                    best_method = method
        
        if best_method:
            report.append(f"Best performing method: {best_method.replace('_', ' ').title()} (R² = {best_r2:.4f})")
        
        # Quality thresholds
        report.append("\nQuality Assessment:")
        high_quality_methods = []
        for method, data in validation_results.items():
            if method != 'consistency_analysis':
                r2 = data.get('r2_score', 0)
                mae = data.get('mae', float('inf'))
                within_2pct = data.get('within_2pct_accuracy', 0)
                
                if r2 > 0.95 and mae < 0.01 and within_2pct > 90:
                    high_quality_methods.append(method)
        
        if high_quality_methods:
            report.append(f"High-quality methods (R² > 0.95, MAE < 0.01, >90% within ±2%): "
                         f"{', '.join([m.replace('_', ' ').title() for m in high_quality_methods])}")
        else:
            report.append("No methods meet all high-quality criteria. Consider method optimization.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

# Example usage and integration functions
def run_complete_validation_pipeline(simulator, simulation_results):
    """
    Run the complete validation pipeline on simulation results
    
    Args:
        simulator: BisulfiteSimulator instance
        simulation_results: Results from simulation
        
    Returns:
        dict: Complete validation results with plots
    """
    print("\n=== Running Complete Validation Pipeline ===")
    
    # Initialize validation framework
    validator = ValidationFramework()
    visualizer = VisualizationSuite()
    
    # Extract true efficiencies and analysis results
    true_efficiencies = list(simulation_results.keys())
    
    # Mock analysis results structure for demonstration
    # In real implementation, this would come from the analyzer
    analysis_results = {}
    
    for efficiency in true_efficiencies:
        # Simulate some analysis results with realistic noise
        noise = np.random.normal(0, 0.005)  # Small amount of noise
        
        analysis_results[efficiency] = {
            'non_cpg_efficiency': {'efficiency': efficiency + noise},
            'lambda_efficiency': {'efficiency': efficiency + noise * 0.8},
            'chh_efficiency': {'efficiency': efficiency + noise * 1.2},
            'confidence_intervals': {'mean_efficiency': efficiency + noise * 0.9},
            'summary_metrics': {
                'consensus_efficiency': efficiency + noise,
                'methods_efficiencies': {
                    'non_cpg': efficiency + noise,
                    'lambda_dna': efficiency + noise * 0.8,
                    'chh_context': efficiency + noise * 1.2
                }
            }
        }
    
    # Run validation
    validation_results = validator.validate_method_accuracy(analysis_results, true_efficiencies)
    
    # Create visualizations
    figures = visualizer.create_accuracy_plots(validation_results)
    
    # Generate report
    report = visualizer.create_comprehensive_report(validation_results)
    
    return {
        'validation_results': validation_results,
        'figures': figures,
        'report': report
    }

if __name__ == "__main__":
    print("Bisulfite Conversion Efficiency Validation and Visualization Module")
    print("This module provides comprehensive validation and visualization tools.")
    print("Use in conjunction with simulation and analysis modules for complete pipeline.")
