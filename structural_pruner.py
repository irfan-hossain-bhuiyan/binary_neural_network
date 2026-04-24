import torch
from discrete_logic_net import DiscreteMultiLayerLogicGateNet
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from binray_transformer import MultiLayerLogicGateNet

def prune_structural_loops(net: DiscreteMultiLayerLogicGateNet):
    """
    Performs structural pruning on the discrete boolean network up to Depth 3.
    It identifies redundant paths (loops) where a variable connects to multiple
    hidden nodes that all route to the same output with identical logic.
    For these redundant connections, it safely sets the initial weight to False.
    """
    layers = net.expectation_layers
    total_pruned = 0
    
    print(f"Starting structural pruning up to Depth 3 on {len(layers)} layers...")
    
    # We check depth 3: L1 -> L2 -> L3 (Input -> Hidden1 -> Hidden2 -> Output)
    for layer_idx in range(len(layers) - 2):
        l1 = layers[layer_idx]
        l2 = layers[layer_idx + 1]
        l3 = layers[layer_idx + 2]
        
        W1, B1 = l1.weight, l1.bias # type: ignore
        W2, B2 = l2.weight, l2.bias # type: ignore
        W3, B3 = l3.weight, l3.bias # type: ignore
        
        pruned_in_layer = 0
        
        # We want to see if connection W1[j1, i] is redundant.
        for j1 in range(W1.shape[0]):
            active_inputs = torch.where(W1[j1])[0].tolist() # type: ignore
            
            for i in active_inputs:
                is_redundant = True
                all_outputs_covered = False
                
                # Get all paths going through this specific W1[j1, i]
                # To Depth 2 (j2)
                active_j2s = torch.where(W2[:, j1])[0].tolist() # type: ignore
                if not active_j2s:
                    continue
                    
                for j2 in active_j2s:
                    if B2[j2, j1]: # type: ignore
                        is_redundant = False
                        break
                        
                    # Output (k)
                    active_outputs = torch.where(W3[:, j2])[0].tolist() # type: ignore
                    if not active_outputs:
                        continue
                        
                    all_outputs_covered = True
                        
                    for k in active_outputs:
                        if B3[k, j2]: # type: ignore
                            is_redundant = False
                            break
                            
                        found_alt_path = False
                        found_tautology = False
                        
                        # Find all j2_alt that connect to k
                        alt_j2s = torch.where(W3[k, :])[0].tolist() # type: ignore
                        
                        for j2_alt in alt_j2s:
                            if j2_alt == j2: continue
                            if B3[k, j2_alt]: continue # type: ignore
                                
                            # Find all j1_alt that connect to j2_alt
                            alt_j1s = torch.where(W2[j2_alt, :])[0].tolist() # type: ignore
                            
                            for j1_alt in alt_j1s:
                                if j1_alt == j1: continue
                                if B2[j2_alt, j1_alt]: continue # type: ignore
                                
                                # Exact duplicate path Check
                                if W1[j1_alt, i] and B1[j1_alt, i] == B1[j1, i]: # type: ignore
                                    found_alt_path = True
                                    break
                                    
                                # Tautology Check (X OR ~X)
                                if W1[j1_alt, i] and B1[j1_alt, i] != B1[j1, i]: # type: ignore
                                    found_tautology = True
                                    break
                                    
                            if found_alt_path or found_tautology: 
                                break
                                
                        if not found_alt_path and not found_tautology:
                            # We couldn't find an alternative path to 'k'. 'j1' is necessary!
                            is_redundant = False
                            break
                    if not is_redundant: break
                        
                if is_redundant and all_outputs_covered:
                    l1.weight[j1, i] = False # type: ignore
                    W1[j1, i] = False # type: ignore
                    pruned_in_layer += 1
                    total_pruned += 1
                    
        print(f"Layer {layer_idx} -> {layer_idx+1} -> {layer_idx+2}: Pruned {pruned_in_layer} redundant weights.")
        
    print(f"\nOptimization Complete! Total structural redundancies removed: {total_pruned}")
    if total_pruned > 0:
        print(f"\n=======================================================")
        print(f"!!! SUCCESS: Squeezed {total_pruned} redundant connections !!!")
        print(f"=======================================================\n")
    return total_pruned

def prune_continuous_network(net: "MultiLayerLogicGateNet", threshold: float = 0.5) -> int:
    """
    Converts a continuous MultiLayerLogicGateNet to its discrete representation,
    applies structural loop pruning to remove redundant connections,
    and maps the pruned, fully discretized weights back into the continuous model.
    """
    print("\n[Pruner] Converting continuous network to discrete...")
    discrete_net = net.to_discrete(threshold=threshold)
    
    total_pruned = prune_structural_loops(discrete_net)
    
    print("[Pruner] Overwriting continuous parameters with pruned discrete logic...")
    with torch.no_grad():
        for cont_layer, disc_layer in zip(net.expectation_layers, discrete_net.expectation_layers):
            cont_layer.weight.data.copy_(disc_layer.weight.to(torch.float32)) # type: ignore
            cont_layer.bias.data.copy_(disc_layer.bias.to(torch.float32)) # type: ignore
            
    return total_pruned

