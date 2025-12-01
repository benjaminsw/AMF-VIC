"""
Enhanced KL Divergence Computation - FULLY FIXED VERSION
All numerical instabilities addressed.
"""

import torch
import numpy as np
from scipy.stats import gaussian_kde
from typing import Tuple, Optional, Union, Dict, Any

''''
def compute_kde_kl_divergence(
    target_samples: torch.Tensor, 
    generated_samples: torch.Tensor,
    grid_resolution: int = 100,
    bandwidth_method: str = 'scott',
    epsilon: float = 1e-10
) -> float:
    """
    Compute KL divergence between target and generated samples using KDE.
    FULLY FIXED: Handles normalization correctly and prevents negative values.
    
    Args:
        target_samples: torch.Tensor [n_samples, 2] - target distribution samples
        generated_samples: torch.Tensor [n_samples, 2] - generated distribution samples  
        grid_resolution: int - number of grid points per dimension
        bandwidth_method: str - bandwidth selection ('scott', 'silverman', or float)
        epsilon: float - small value to avoid log(0) issues
        
    Returns:
        float - KL divergence D_KL(P||Q) where P=target, Q=generated (guaranteed >= 0)
    """
    # Convert to numpy and transpose for KDE
    target_np = target_samples.detach().cpu().numpy().T
    generated_np = generated_samples.detach().cpu().numpy().T
    
    # Fit KDE to both distributions independently
    kde_target = gaussian_kde(target_np, bw_method=bandwidth_method)
    kde_generated = gaussian_kde(generated_np, bw_method=bandwidth_method)
    
    # Create evaluation grid with larger margins
    x_min = min(target_np[0].min(), generated_np[0].min())
    x_max = max(target_np[0].max(), generated_np[0].max())
    y_min = min(target_np[1].min(), generated_np[1].min())
    y_max = max(target_np[1].max(), generated_np[1].max())
    
    # Use 25% margins for better coverage
    x_margin = (x_max - x_min) * 0.25
    y_margin = (y_max - y_min) * 0.25
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    
    # Create grid
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    
    # Evaluate densities on grid
    p_densities = kde_target(grid_points)
    q_densities = kde_generated(grid_points)
    
    # Cell area for numerical integration
    dx = (x_max - x_min) / (grid_resolution - 1)
    dy = (y_max - y_min) / (grid_resolution - 1)
    cell_area = dx * dy
    
    # Normalize to create discrete probability distributions
    p_sum = np.sum(p_densities) * cell_area
    q_sum = np.sum(q_densities) * cell_area
    
    # Safeguard against division by zero
    if p_sum < 1e-15 or q_sum < 1e-15:
        return float('inf')  # Distributions don't overlap on grid
    
    p_probs = (p_densities * cell_area) / p_sum
    q_probs = (q_densities * cell_area) / q_sum
    
    # Add epsilon and re-normalize
    p_probs = np.maximum(p_probs, epsilon)
    q_probs = np.maximum(q_probs, epsilon)
    p_probs = p_probs / np.sum(p_probs)
    q_probs = q_probs / np.sum(q_probs)
    
    # Compute KL divergence with numerical safeguards
    # Only include terms where p_probs is significant
    mask = p_probs > 1e-10
    if not np.any(mask):
        return 0.0
    
    log_ratio = np.log(p_probs[mask] / q_probs[mask])
    kl_divergence = np.sum(p_probs[mask] * log_ratio)
    
    # Final safeguard: ensure non-negative (clip tiny negatives from numerical error)
    kl_divergence = max(0.0, kl_divergence)
    
    return float(kl_divergence)
'''

def compute_kde_kl_divergence(
    target_samples: torch.Tensor, 
    generated_samples: torch.Tensor,
    grid_resolution: int = 100,  # unused, kept for compatibility
    bandwidth_method: str = 'scott',
    epsilon: float = 1e-10
) -> float:
    """Monte Carlo KDE-based KL divergence (handles extreme scale differences)"""
    target_np = target_samples.detach().cpu().numpy().T
    generated_np = generated_samples.detach().cpu().numpy().T
    
    kde_target = gaussian_kde(target_np, bw_method=bandwidth_method)
    kde_generated = gaussian_kde(generated_np, bw_method=bandwidth_method)
    
    # Monte Carlo: evaluate at target samples
    log_p = np.log(kde_target(target_np) + epsilon)
    log_q = np.log(kde_generated(target_np) + epsilon)
    kl_divergence = np.mean(log_p - log_q)
    
    return max(0.0, float(kl_divergence))

def compute_exact_cross_entropy(target_samples: torch.Tensor, 
                               flow_model: torch.nn.Module) -> float:
    """
    Compute exact cross-entropy surrogate: -E_p[log q(x)]
    """
    flow_model.eval()
    with torch.no_grad():
        log_q_x = flow_model.log_prob(target_samples)
        cross_entropy = -log_q_x.mean().item()
    return cross_entropy


def compute_exact_kl_divergence_nf_to_nf(flow_p: torch.nn.Module,
                                        flow_q: torch.nn.Module,
                                        n_samples: int = 10000) -> float:
    """
    Compute exact KL divergence between two normalizing flows: KL(P||Q)
    """
    flow_p.eval()
    flow_q.eval()
    
    with torch.no_grad():
        x_samples = flow_p.sample(n_samples)
        log_p_x = flow_p.log_prob(x_samples)
        log_q_x = flow_q.log_prob(x_samples)
        kl_divergence = (log_p_x - log_q_x).mean().item()
        
    return max(0.0, kl_divergence)  # Safeguard


def compute_kde_cross_entropy(target_samples: torch.Tensor,
                             generated_samples: torch.Tensor,
                             grid_resolution: int = 100,
                             bandwidth_method: str = 'scott',
                             epsilon: float = 1e-10) -> float:
    """
    Compute cross-entropy using KDE: -E_p[log q(x)]
    """
    target_np = target_samples.detach().cpu().numpy().T
    generated_np = generated_samples.detach().cpu().numpy().T
    
    kde_target = gaussian_kde(target_np, bw_method=bandwidth_method)
    kde_generated = gaussian_kde(generated_np, bw_method=bandwidth_method)
    
    # Create evaluation grid
    x_min = min(target_np[0].min(), generated_np[0].min())
    x_max = max(target_np[0].max(), generated_np[0].max())
    y_min = min(target_np[1].min(), generated_np[1].min())
    y_max = max(target_np[1].max(), generated_np[1].max())
    
    x_margin = (x_max - x_min) * 0.2
    y_margin = (y_max - y_min) * 0.2
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    
    p_densities = kde_target(grid_points)
    q_densities = kde_generated(grid_points)
    
    dx = (x_max - x_min) / (grid_resolution - 1)
    dy = (y_max - y_min) / (grid_resolution - 1)
    cell_area = dx * dy
    
    # Normalize
    p_sum = np.sum(p_densities) * cell_area
    q_sum = np.sum(q_densities) * cell_area
    
    if p_sum < 1e-15 or q_sum < 1e-15:
        return float('inf')
    
    p_probs = (p_densities * cell_area) / p_sum
    q_probs = (q_densities * cell_area) / q_sum
    
    q_probs = np.maximum(q_probs, epsilon)
    q_probs = q_probs / np.sum(q_probs)
    p_probs = p_probs / np.sum(p_probs)
    
    cross_entropy = -np.sum(p_probs * np.log(q_probs))
    
    return float(cross_entropy)


def comprehensive_divergence_computation(target_samples: torch.Tensor,
                                       generated_samples: torch.Tensor,
                                       flow_model: Optional[torch.nn.Module] = None,
                                       target_flow: Optional[torch.nn.Module] = None,
                                       n_samples_exact: int = 10000,
                                       grid_resolution: int = 100,
                                       bandwidth_method: str = 'scott') -> Dict[str, float]:
    """
    Comprehensive computation of all available divergence metrics.
    """
    results = {}
    
    # KDE-based methods
    results['kde_kl_divergence'] = compute_kde_kl_divergence(
        target_samples, generated_samples, grid_resolution, bandwidth_method)
    
    results['kde_cross_entropy'] = compute_kde_cross_entropy(
        target_samples, generated_samples, grid_resolution, bandwidth_method)
    
    # Exact cross-entropy
    if flow_model is not None:
        results['exact_cross_entropy'] = compute_exact_cross_entropy(
            target_samples, flow_model)
    
    # Exact KL divergence
    if target_flow is not None and flow_model is not None:
        results['exact_kl_divergence'] = compute_exact_kl_divergence_nf_to_nf(
            target_flow, flow_model, n_samples_exact)
    
    # Comparisons
    if flow_model is not None:
        exact_ce = results.get('exact_cross_entropy')
        kde_ce = results.get('kde_cross_entropy')
        if exact_ce is not None and kde_ce is not None:
            results['cross_entropy_difference'] = abs(exact_ce - kde_ce)
            results['cross_entropy_relative_error'] = abs(exact_ce - kde_ce) / max(abs(exact_ce), 1e-10)
    
    if 'exact_kl_divergence' in results:
        exact_kl = results['exact_kl_divergence']
        kde_kl = results['kde_kl_divergence']
        results['kl_difference'] = abs(exact_kl - kde_kl)
        results['kl_relative_error'] = abs(exact_kl - kde_kl) / max(abs(exact_kl), 1e-10)
    
    return results


def adaptive_grid_resolution(target_samples: torch.Tensor,
                           generated_samples: torch.Tensor,
                           max_resolution: int = 200,
                           min_resolution: int = 50) -> int:
    """
    Adaptively choose grid resolution based on data characteristics.
    """
    n_target = target_samples.size(0)
    n_generated = generated_samples.size(0)
    total_samples = n_target + n_generated
    
    if total_samples < 500:
        return min_resolution
    elif total_samples < 2000:
        return 100
    else:
        return max_resolution


def get_kde_info(target_samples: torch.Tensor, 
                generated_samples: torch.Tensor) -> dict:
    """
    Get information about KDE fitting for debugging.
    """
    target_np = target_samples.detach().cpu().numpy().T
    generated_np = generated_samples.detach().cpu().numpy().T
    
    kde_target = gaussian_kde(target_np, bw_method='scott')
    kde_generated = gaussian_kde(generated_np, bw_method='scott')
    
    return {
        'target_bandwidth': kde_target.factor,
        'generated_bandwidth': kde_generated.factor,
        'target_n_samples': target_samples.size(0),
        'generated_n_samples': generated_samples.size(0),
        'target_mean': target_samples.mean(dim=0).tolist(),
        'generated_mean': generated_samples.mean(dim=0).tolist(),
        'target_std': target_samples.std(dim=0).tolist(),
        'generated_std': generated_samples.std(dim=0).tolist(),
    }