import torch
import numpy as np
import ot
from threeflows_amf_vi_weights_log import SequentialAMFVI, train_sequential_amf_vi
from data.data_generator import generate_data
import os
import pickle
import csv

# Set seed for reproducible experiments
torch.manual_seed(2025)
np.random.seed(2025)

def compute_sinkhorn_divergence(target_samples, generated_samples, reg=0.1, max_iter=1000):
    """
    Compute Sinkhorn divergence between two sets of samples.
    
    Sinkhorn divergence: S_ε(μ,ν) = OT_ε(μ,ν) - 0.5[OT_ε(μ,μ) + OT_ε(ν,ν)]
    where OT_ε is the entropy-regularized optimal transport cost.
    """
    target_np = target_samples.detach().cpu().numpy()
    generated_np = generated_samples.detach().cpu().numpy()
    
    # Create uniform distributions
    a = np.ones(len(target_np)) / len(target_np)
    b = np.ones(len(generated_np)) / len(generated_np)
    
    # Compute cost matrices (squared L2 distances)
    C_xy = ot.dist(target_np, generated_np, metric='sqeuclidean')
    C_xx = ot.dist(target_np, target_np, metric='sqeuclidean')
    C_yy = ot.dist(generated_np, generated_np, metric='sqeuclidean')
    
    # Compute entropy-regularized optimal transport costs
    OT_xy = ot.sinkhorn2(a, b, C_xy, reg, numItermax=max_iter)
    OT_xx = ot.sinkhorn2(a, a, C_xx, reg, numItermax=max_iter)
    OT_yy = ot.sinkhorn2(b, b, C_yy, reg, numItermax=max_iter)
    
    # Compute Sinkhorn divergence
    sinkhorn_div = OT_xy - 0.5 * (OT_xx + OT_yy)
    
    return sinkhorn_div

def compute_sliced_sinkhorn_divergence(target_samples, generated_samples, n_projections=100, reg=0.1):
    """Compute Sliced Sinkhorn Divergence using random projections."""
    target_np = target_samples.detach().cpu().numpy()
    generated_np = generated_samples.detach().cpu().numpy()
    
    # Generate random unit vectors for projections
    directions = np.random.randn(n_projections, target_np.shape[1])
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    sinkhorn_divergences = []
    for direction in directions:
        # Project samples onto direction
        proj_target = target_np @ direction
        proj_generated = generated_np @ direction
        
        # Reshape for 1D optimal transport
        proj_target = proj_target.reshape(-1, 1)
        proj_generated = proj_generated.reshape(-1, 1)
        
        # Convert back to torch tensors for sinkhorn computation
        proj_target_tensor = torch.from_numpy(proj_target).float()
        proj_generated_tensor = torch.from_numpy(proj_generated).float()
        
        # Compute 1D Sinkhorn divergence
        sinkhorn_div_1d = compute_sinkhorn_divergence(proj_target_tensor, proj_generated_tensor, reg)
        sinkhorn_divergences.append(sinkhorn_div_1d)
    
    return np.mean(sinkhorn_divergences)

def compute_sinkhorn_distance(target_samples, flow_model, metric_type='sinkhorn', reg=0.1):
    """Compute Sinkhorn divergence/distance between target and flow-generated samples."""
    with torch.no_grad():
        generated_samples = flow_model.sample(1000)
        
        if metric_type == 'sinkhorn':
            return compute_sinkhorn_divergence(target_samples, generated_samples, reg)
        elif metric_type == 'sliced_sinkhorn':
            return compute_sliced_sinkhorn_divergence(target_samples, generated_samples, reg=reg)
        else:
            raise ValueError("metric_type must be 'sinkhorn' or 'sliced_sinkhorn'")

def evaluate_individual_flows_sinkhorn(model, test_data, flow_names, dataset_name, reg=0.1):
    """Evaluate each individual flow using both Sinkhorn divergence metrics."""
    individual_metrics = {}
    
    with torch.no_grad():
        for i, (flow, name) in enumerate(zip(model.flows, flow_names)):
            # Compute both regular and sliced Sinkhorn divergence
            sinkhorn_div = compute_sinkhorn_distance(test_data, flow, 'sinkhorn', reg)
            sliced_sinkhorn_div = compute_sinkhorn_distance(test_data, flow, 'sliced_sinkhorn', reg)
            
            individual_metrics[name] = {
                'sinkhorn_divergence': sinkhorn_div,
                'sliced_sinkhorn_divergence': sliced_sinkhorn_div,
            }
    
    return individual_metrics

def evaluate_single_sequential_dataset_sinkhorn(dataset_name, reg=0.1):
    """Evaluate a single Sequential model using Sinkhorn divergence."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating Sinkhorn Divergence for {dataset_name.upper()} dataset")
    print(f"Regularization parameter: {reg}")
    print(f"{'='*50}")
    
    # Create test data
    test_data = generate_data(dataset_name, n_samples=1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = test_data.to(device)
    
    # Load or train model
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']
    else:
        print(f"Training new model for {dataset_name}")
        model, _, _ = train_sequential_amf_vi(dataset_name, show_plots=False, save_plots=False)
        
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'dataset': dataset_name}, f)
    
    model = model.to(device)
    
    # Get flow names
    flow_type_map = {
        'RealNVPFlow': 'realnvp', 'MAFFlow': 'maf', 'NAFFlowSimplified': 'naf',
        'NICEFlow': 'nice', 'IAFFlow': 'iaf', 'GaussianizationFlow': 'gaussianization',
        'GlowFlow': 'glow', 'TANFlow': 'tan', 'RBIGFlow': 'rbig'
    }
    
    flow_names = [flow_type_map.get(flow.__class__.__name__, flow.__class__.__name__.lower()) 
                  for flow in model.flows]
    
    # Compute mixture model Sinkhorn divergences
    model.eval()
    mixture_sinkhorn_div = compute_sinkhorn_distance(test_data, model, 'sinkhorn', reg)
    mixture_sliced_sinkhorn_div = compute_sinkhorn_distance(test_data, model, 'sliced_sinkhorn', reg)
    
    # Evaluate individual flows
    individual_flow_metrics = evaluate_individual_flows_sinkhorn(model, test_data, flow_names, dataset_name, reg)
    
    # Get learned weights
    if hasattr(model, 'weights_trained') and model.weights_trained:
        if hasattr(model, 'log_weights'):
            learned_weights = torch.softmax(model.log_weights, dim=0).detach().cpu().numpy()
        else:
            learned_weights = model.weights.detach().cpu().numpy()
    else:
        learned_weights = np.ones(len(model.flows)) / len(model.flows)
    
    results = {
        'dataset': dataset_name,
        'regularization': reg,
        'mixture_sinkhorn_divergence': mixture_sinkhorn_div,
        'mixture_sliced_sinkhorn_divergence': mixture_sliced_sinkhorn_div,
        'individual_flow_metrics': individual_flow_metrics,
        'learned_weights': learned_weights,
        'weights_trained': getattr(model, 'weights_trained', False),
        'flow_names': flow_names
    }
    
    print(f"📊 Sinkhorn Results for {dataset_name}:")
    print(f"   Mixture Sinkhorn Divergence: {mixture_sinkhorn_div:.4f}")
    print(f"   Mixture Sliced Sinkhorn Divergence: {mixture_sliced_sinkhorn_div:.4f}")
    print(f"   Learned Weights: {learned_weights}")
    
    return results

def comprehensive_sinkhorn_evaluation(reg_values=[0.01, 0.1, 1.0]):
    """Comprehensive Sinkhorn divergence evaluation of all datasets with multiple regularization parameters."""
    
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 'multimodal', 'two_moons', 'rings']
    all_results = {}
    
    # Evaluate each dataset with different regularization parameters
    for reg in reg_values:
        print(f"\n{'='*60}")
        print(f"EVALUATING WITH REGULARIZATION ε = {reg}")
        print(f"{'='*60}")
        
        reg_results = {}
        for dataset_name in datasets:
            try:
                results = evaluate_single_sequential_dataset_sinkhorn(dataset_name, reg)
                if results is not None:
                    reg_results[dataset_name] = results
            except Exception as e:
                print(f"❌ Failed to evaluate {dataset_name} with reg={reg}: {e}")
                continue
        
        if reg_results:
            all_results[f'reg_{reg}'] = reg_results
    
    if not all_results:
        print("❌ No models could be evaluated.")
        return None
    
    # Create CSV data for all regularization parameters
    summary_data = []
    for reg_key, reg_results in all_results.items():
        reg_value = float(reg_key.split('_')[1])
        
        for dataset_name, results in reg_results.items():
            weights_status = "Yes" if results['weights_trained'] else "No"
            
            for i, flow_name in enumerate(results['flow_names']):
                individual_metrics = results['individual_flow_metrics'].get(flow_name, {})
                flow_sinkhorn_div = individual_metrics.get('sinkhorn_divergence', 0.0)
                flow_sliced_sinkhorn_div = individual_metrics.get('sliced_sinkhorn_divergence', 0.0)
                flow_weight = results['learned_weights'][i]
                
                summary_data.append([
                    dataset_name,
                    reg_value,
                    results['mixture_sinkhorn_divergence'],
                    results['mixture_sliced_sinkhorn_divergence'],
                    flow_name.upper(),
                    flow_sinkhorn_div,
                    flow_sliced_sinkhorn_div,
                    flow_weight,
                    weights_status
                ])
    
    # Save to CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'sinkhorn_comprehensive_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'regularization', 'mixture_sinkhorn_divergence', 'mixture_sliced_sinkhorn_divergence',
                        'flow', 'flow_sinkhorn_divergence', 'flow_sliced_sinkhorn_divergence', 
                        'flow_weight', 'weights_trained'])
        writer.writerows(summary_data)
        print('✅ sinkhorn_comprehensive_metrics.csv successfully created')
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    for reg_key, reg_results in all_results.items():
        reg_value = float(reg_key.split('_')[1])
        print(f"\nRegularization ε = {reg_value}:")
        
        mixture_sinkhorn_values = [r['mixture_sinkhorn_divergence'] for r in reg_results.values()]
        mixture_sliced_values = [r['mixture_sliced_sinkhorn_divergence'] for r in reg_results.values()]
        
        print(f"  Mixture Sinkhorn Divergence - Mean: {np.mean(mixture_sinkhorn_values):.4f}, Std: {np.std(mixture_sinkhorn_values):.4f}")
        print(f"  Mixture Sliced Sinkhorn Divergence - Mean: {np.mean(mixture_sliced_values):.4f}, Std: {np.std(mixture_sliced_values):.4f}")
    
    return all_results

if __name__ == "__main__":
    # Run evaluation with multiple regularization parameters
    results = comprehensive_sinkhorn_evaluation(reg_values=[0.01, 0.1, 1.0])
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print("Results saved to: results/sinkhorn_comprehensive_metrics.csv")
    print(f"{'='*60}")