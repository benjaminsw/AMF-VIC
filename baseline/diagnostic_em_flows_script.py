"""
Comprehensive diagnostic script for EM Mixture of Flows.
Checks training failure, saved model state, and component health.
"""

import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from pathlib import Path


def diagnose_em_mixture_model(dataset_name='two_moons', model_dir='baseline_results'):
    """
    Comprehensive diagnostics for EM Mixture model training and state.
    
    Args:
        dataset_name: Name of the dataset to diagnose
        model_dir: Directory containing the saved model
    """
    
    print("="*80)
    print(f"COMPREHENSIVE DIAGNOSTIC REPORT: {dataset_name.upper()}")
    print("="*80)
    
    # Load the model
    model_path = os.path.join(model_dir, f'em_mixture_{dataset_name}.pkl')
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return None
    
    print(f"\nLoading model from: {model_path}")
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    model = saved_data['model']
    losses = saved_data.get('losses', [])
    weight_history = saved_data.get('weight_history', [])
    
    print(f"Model loaded successfully!")
    print(f"  Number of components: {model.K}")
    print(f"  Dimension: {model.dim}")
    print(f"  Training epochs: {len(losses)}")
    
    # ============================================================
    # 1. WHY DID TRAINING FAIL?
    # ============================================================
    print("\n" + "="*80)
    print("1. TRAINING FAILURE ANALYSIS")
    print("="*80)
    
    if not losses:
        print("  WARNING: No loss history found!")
    else:
        print(f"\n  Loss Statistics:")
        print(f"    Initial loss: {losses[0]:.6f}")
        print(f"    Final loss: {losses[-1]:.6f}")
        print(f"    Best loss: {min(losses):.6f} (epoch {losses.index(min(losses))})")
        print(f"    Worst loss: {max(losses):.6f} (epoch {losses.index(max(losses))})")
        print(f"    Loss improvement: {losses[0] - losses[-1]:.6f}")
        
        # Check for NaN or Inf in losses
        nan_count = sum(1 for l in losses if np.isnan(l) or np.isinf(l))
        if nan_count > 0:
            print(f"    ⚠️  WARNING: {nan_count} epochs with NaN/Inf losses!")
        
        # Check convergence
        last_100 = losses[-100:] if len(losses) >= 100 else losses
        loss_variance = np.var(last_100)
        print(f"    Loss variance (last 100 epochs): {loss_variance:.6f}")
        
        if loss_variance < 1e-6:
            print(f"    ✓ Model converged (low variance)")
        else:
            print(f"    ⚠️  Model may not have converged (high variance)")
        
        # Check for divergence
        if len(losses) > 10:
            early_mean = np.mean(losses[:10])
            late_mean = np.mean(losses[-10:])
            if late_mean > early_mean + 1.0:
                print(f"    ⚠️  DIVERGENCE DETECTED: Loss increased from {early_mean:.4f} to {late_mean:.4f}")
        
        # Show loss trend
        print(f"\n  Loss Trend (showing every 500 epochs):")
        step = max(1, len(losses) // 10)
        for i in range(0, len(losses), step):
            print(f"    Epoch {i:5d}: {losses[i]:.6f}")
        print(f"    Epoch {len(losses)-1:5d}: {losses[-1]:.6f}")
    
    # ============================================================
    # 2. INSPECT THE SAVED MODEL
    # ============================================================
    print("\n" + "="*80)
    print("2. SAVED MODEL INSPECTION")
    print("="*80)
    
    print(f"\n  Mixture Weights:")
    print(f"    Current weights: {model.mixture_weights.cpu().numpy()}")
    print(f"    Sum: {model.mixture_weights.sum().item():.6f}")
    print(f"    Min weight: {model.mixture_weights.min().item():.6f}")
    print(f"    Max weight: {model.mixture_weights.max().item():.6f}")
    
    # Check for degenerate weights
    for k, weight in enumerate(model.mixture_weights.cpu().numpy()):
        if weight < 0.05:
            print(f"    ⚠️  Component {k} is nearly dead (weight={weight:.4f})")
        elif weight > 0.8:
            print(f"    ⚠️  Component {k} dominates (weight={weight:.4f})")
    
    # Test each flow individually
    print(f"\n  Individual Flow Diagnostics:")
    device = model.device
    
    for k, flow in enumerate(model.flows):
        print(f"\n    --- Flow {k} (weight={model.mixture_weights[k].item():.4f}) ---")
        
        # Check parameters
        param_count = sum(p.numel() for p in flow.parameters())
        print(f"      Parameters: {param_count:,}")
        
        # Check for NaN/Inf in parameters
        has_nan = any(torch.isnan(p).any().item() for p in flow.parameters())
        has_inf = any(torch.isinf(p).any().item() for p in flow.parameters())
        
        if has_nan:
            print(f"      ⚠️  CRITICAL: Parameters contain NaN!")
        if has_inf:
            print(f"      ⚠️  CRITICAL: Parameters contain Inf!")
        
        if not has_nan and not has_inf:
            print(f"      ✓ Parameters are finite")
        
        # Test forward pass with standard Gaussian input
        flow.eval()
        with torch.no_grad():
            z_test = torch.randn(1000, model.dim, device=device)
            try:
                x_test, log_det_test = flow(z_test)
                
                print(f"      Output statistics:")
                print(f"        Mean: [{x_test.mean(0)[0].item():.4f}, {x_test.mean(0)[1].item():.4f}]")
                print(f"        Std:  [{x_test.std(0)[0].item():.4f}, {x_test.std(0)[1].item():.4f}]")
                print(f"        Min:  {x_test.min().item():.4f}")
                print(f"        Max:  {x_test.max().item():.4f}")
                print(f"        Range: {x_test.max().item() - x_test.min().item():.4f}")
                
                # Check for abnormal outputs
                if torch.isnan(x_test).any():
                    print(f"        ⚠️  CRITICAL: Output contains NaN!")
                elif torch.isinf(x_test).any():
                    print(f"        ⚠️  CRITICAL: Output contains Inf!")
                elif abs(x_test.mean().item()) > 100:
                    print(f"        ⚠️  WARNING: Mean is very large (diverged?)")
                elif x_test.std().item() > 100:
                    print(f"        ⚠️  WARNING: Std is very large (diverged?)")
                else:
                    print(f"        ✓ Output looks reasonable")
                
                # Check log determinant
                print(f"      Log determinant statistics:")
                print(f"        Mean: {log_det_test.mean().item():.4f}")
                print(f"        Std:  {log_det_test.std().item():.4f}")
                print(f"        Range: [{log_det_test.min().item():.4f}, {log_det_test.max().item():.4f}]")
                
                if torch.isnan(log_det_test).any() or torch.isinf(log_det_test).any():
                    print(f"        ⚠️  Log determinant contains NaN/Inf!")
                
            except Exception as e:
                print(f"        ⚠️  CRITICAL: Forward pass failed with error: {e}")
    
    # ============================================================
    # 3. CHECK TRAINING PARAMETERS
    # ============================================================
    print("\n" + "="*80)
    print("3. TRAINING PARAMETERS ANALYSIS")
    print("="*80)
    
    print(f"\n  Model Architecture:")
    print(f"    Number of components (K): {model.K}")
    print(f"    Data dimension: {model.dim}")
    print(f"    Flow layers per component: {len(model.flows[0].layers)}")
    print(f"    Total parameters: {sum(sum(p.numel() for p in flow.parameters()) for flow in model.flows):,}")
    
    print(f"\n  Training Configuration:")
    print(f"    Number of epochs: {len(losses)}")
    print(f"    Learning rate: {model.optimizers[0].param_groups[0]['lr']}")
    
    # Check if learning rate is reasonable
    lr = model.optimizers[0].param_groups[0]['lr']
    if lr > 1e-2:
        print(f"    ⚠️  Learning rate may be too high ({lr})")
    elif lr < 1e-5:
        print(f"    ⚠️  Learning rate may be too low ({lr})")
    else:
        print(f"    ✓ Learning rate seems reasonable")
    
    # ============================================================
    # 4. VERIFY ONE COMPONENT IS DEAD
    # ============================================================
    print("\n" + "="*80)
    print("4. COMPONENT HEALTH CHECK")
    print("="*80)
    
    weights_np = model.mixture_weights.cpu().numpy()
    
    print(f"\n  Component Weight Distribution:")
    for k, weight in enumerate(weights_np):
        bar_length = int(weight * 50)
        bar = "█" * bar_length
        status = ""
        
        if weight < 0.01:
            status = " ⚠️  DEAD"
        elif weight < 0.05:
            status = " ⚠️  DYING"
        elif weight < 0.15:
            status = " ⚠️  WEAK"
        elif weight > 0.7:
            status = " ⚠️  DOMINANT"
        else:
            status = " ✓ HEALTHY"
        
        print(f"    Component {k}: {bar} {weight:.4f}{status}")
    
    # Analyze weight evolution
    if weight_history:
        print(f"\n  Weight Evolution Analysis:")
        print(f"    Total epochs tracked: {len(weight_history)}")
        
        # Show weight trajectory for each component
        print(f"\n    Weight trajectory (showing every 1000 epochs):")
        step = max(1, len(weight_history) // 10)
        for i in range(0, len(weight_history), step):
            weights_str = ", ".join([f"{w:.4f}" for w in weight_history[i]])
            print(f"      Epoch {i:5d}: [{weights_str}]")
        
        # Final weights
        final_weights_str = ", ".join([f"{w:.4f}" for w in weight_history[-1]])
        print(f"      Epoch {len(weight_history)-1:5d}: [{final_weights_str}]")
        
        # Check for collapse
        weight_history_np = np.array(weight_history)
        for k in range(model.K):
            component_weights = weight_history_np[:, k]
            initial_weight = component_weights[0]
            final_weight = component_weights[-1]
            
            if initial_weight > 0.2 and final_weight < 0.05:
                print(f"\n    ⚠️  Component {k} COLLAPSED during training!")
                print(f"       Started at {initial_weight:.4f}, ended at {final_weight:.4f}")
    
    # ============================================================
    # 5. VISUALIZATIONS
    # ============================================================
    print("\n" + "="*80)
    print("5. GENERATING DIAGNOSTIC PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss curve
    if losses:
        axes[0, 0].plot(losses, linewidth=1, color='darkblue', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Negative Log-Likelihood')
        axes[0, 0].set_title('Training Loss Curve')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Highlight potential issues
        if len(losses) > 100:
            window = 100
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(losses)), moving_avg, 
                          color='red', linewidth=2, label='Moving Average')
            axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No loss history available', 
                       ha='center', va='center', fontsize=12)
    
    # Plot 2: Weight evolution
    if weight_history:
        weight_history_np = np.array(weight_history)
        for k in range(model.K):
            axes[0, 1].plot(weight_history_np[:, k], 
                          label=f'Component {k}', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Mixture Weight')
        axes[0, 1].set_title('Component Weight Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
    else:
        axes[0, 1].text(0.5, 0.5, 'No weight history available', 
                       ha='center', va='center', fontsize=12)
    
    # Plot 3: Final weight distribution (bar chart)
    axes[1, 0].bar(range(model.K), weights_np, color=['green', 'orange', 'blue'])
    axes[1, 0].set_xlabel('Component')
    axes[1, 0].set_ylabel('Mixture Weight')
    axes[1, 0].set_title('Final Component Weights')
    axes[1, 0].set_xticks(range(model.K))
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axhline(y=1/model.K, color='red', linestyle='--', 
                      label=f'Uniform ({1/model.K:.3f})')
    axes[1, 0].legend()
    
    # Plot 4: Sample generation test
    try:
        with torch.no_grad():
            samples = model.sample(1000)
            samples_np = samples.cpu().numpy()
            
        axes[1, 1].scatter(samples_np[:, 0], samples_np[:, 1], 
                         alpha=0.5, s=10, color='purple')
        axes[1, 1].set_xlabel('Dimension 1')
        axes[1, 1].set_ylabel('Dimension 2')
        axes[1, 1].set_title('Generated Samples (1000)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Mean: [{samples_np.mean(0)[0]:.2f}, {samples_np.mean(0)[1]:.2f}]\n"
        stats_text += f"Std: [{samples_np.std(0)[0]:.2f}, {samples_np.std(0)[1]:.2f}]"
        axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.5), fontsize=9)
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, f'Sample generation failed:\n{str(e)}', 
                       ha='center', va='center', fontsize=10, color='red')
    
    plt.suptitle(f'EM Mixture Diagnostic Report: {dataset_name}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(model_dir, f'diagnostic_{dataset_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n  Diagnostic plots saved to: {plot_path}")
    plt.close()
    
    # ============================================================
    # 6. SUMMARY AND RECOMMENDATIONS
    # ============================================================
    print("\n" + "="*80)
    print("6. SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    issues = []
    recommendations = []
    
    # Check for issues
    if losses and losses[-1] > losses[0]:
        issues.append("Training diverged (loss increased)")
        recommendations.append("Reduce learning rate (try 1e-4 instead of 1e-3)")
    
    if any(w < 0.05 for w in weights_np):
        issues.append("One or more components collapsed")
        recommendations.append("Try fewer components (K=2 instead of K=3)")
        recommendations.append("Initialize with better starting weights")
    
    if any(w > 0.8 for w in weights_np):
        issues.append("One component dominates")
        recommendations.append("Model may need only 1 component for this dataset")
    
    # Check flow outputs
    try:
        with torch.no_grad():
            test_samples = model.sample(100)
            if test_samples.abs().max() > 1000:
                issues.append("Generated samples have extreme values")
                recommendations.append("Flow transformations diverged - reduce learning rate")
                recommendations.append("Add gradient clipping during training")
    except:
        issues.append("Cannot generate samples")
        recommendations.append("Model is completely broken - retrain from scratch")
    
    if issues:
        print("\n  ⚠️  Issues Detected:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        
        print("\n  💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec}")
    else:
        print("\n  ✓ No major issues detected!")
        print("  Model appears to have trained successfully.")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC REPORT COMPLETE")
    print("="*80)
    
    return {
        'model': model,
        'losses': losses,
        'weight_history': weight_history,
        'issues': issues,
        'recommendations': recommendations
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Get dataset name from command line or use default
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'two_moons'
    
    # Determine model directory
    if len(sys.argv) > 2:
        model_dir = sys.argv[2]
    else:
        # Auto-detect: look in baseline/baseline_results relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'baseline_results')
        
        # If not found, try parent directory structure
        if not os.path.exists(model_dir):
            parent_dir = os.path.dirname(script_dir)
            model_dir = os.path.join(parent_dir, 'baseline', 'baseline_results')
    
    print(f"Running diagnostics for dataset: {dataset}")
    print(f"Looking for models in: {model_dir}")
    
    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"\n✗ ERROR: Directory not found: {model_dir}")
        print(f"\nPlease specify the correct path:")
        print(f"  python {os.path.basename(__file__)} {dataset} /path/to/baseline_results")
        sys.exit(1)
    
    # Check if model file exists
    model_file = os.path.join(model_dir, f'em_mixture_{dataset}.pkl')
    if not os.path.exists(model_file):
        print(f"\n✗ ERROR: Model file not found: {model_file}")
        print(f"\nAvailable models in {model_dir}:")
        try:
            files = [f for f in os.listdir(model_dir) if f.startswith('em_mixture_') and f.endswith('.pkl')]
            if files:
                for f in sorted(files):
                    print(f"  - {f}")
            else:
                print("  (no EM mixture models found)")
        except:
            print("  (could not list directory)")
        sys.exit(1)
    
    print(f"Model file found: {model_file}\n")
    
    results = diagnose_em_mixture_model(dataset, model_dir)
    
    if results:
        print("\n✓ Diagnostic complete! Check the generated plots and recommendations above.")
    else:
        print("\n✗ Diagnostic failed - check error messages above.")
