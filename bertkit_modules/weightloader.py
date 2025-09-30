import torch
import torch.nn as nn
import os
from typing import Dict, Any, Optional, List

class WeightLoader:
    """Utility class for loading PyTorch weights from .bin files into models"""
    
    def __init__(self, strict=False, verbose=True):
        """
        Args:
            strict: If True, raise error on mismatched keys
            verbose: If True, print loading information
        """
        self.strict = strict
        self.verbose = verbose
    
    def load_pytorch_bin(self, bin_path: str) -> Dict[str, torch.Tensor]:
        """
        Load weights from a pytorch_model.bin file. By inspection it is sometimes found
        that a prefix of the layer names (such as bert.) has to be removed from the state_dict
        keys in order to produce fully matching keys. For this reason the class provides an
        internal function _apply_prefix_mapping.
        
        Args:
            bin_path: Path to the .bin file
            
        Returns:
            Dictionary of parameter names to tensors
        """
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Weight file not found: {bin_path}")
        
        if self.verbose:
            print(f"Loading weights from: {bin_path}")
        
        # Load the state dict
        state_dict = torch.load(bin_path, map_location='cpu')
        
        if self.verbose:
            print(f"Found {len(state_dict)} parameters in weight file")
            # Show some parameter info
            for i, (name, tensor) in enumerate(state_dict.items()):
                print(f"  {name}: {tensor.shape} ({tensor.dtype})")
                if i >= 5:  # Show first 5 parameters
                    print(f"  ... and {len(state_dict) - 6} more parameters")
                    break
        
        return state_dict
    

    def _apply_prefix_mapping(self, state_dict: Dict[str, torch.Tensor], 
                             prefix_mapping: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Apply prefix mapping to parameter names"""
        new_state_dict = {}
        
        for old_key, tensor in state_dict.items():
            new_key = old_key
            
            # Apply prefix mappings
            for old_prefix, new_prefix in prefix_mapping.items():
                if old_key.startswith(old_prefix):
                    new_key = old_key.replace(old_prefix, new_prefix, 1)
                    break
            
            new_state_dict[new_key] = tensor
        
        return new_state_dict
    
    
    def _map_layernorm_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply correct mapping of layernorms"""
        
        new_state_dict = {}
        
        for old_key, tensor in state_dict.items():
            
            new_key = old_key
            
            if 'gamma' in new_key:
                new_key = new_key.replace('gamma', 'weight')
            elif 'beta' in new_key:
                new_key = new_key.replace('beta', 'bias')
            
            new_state_dict[new_key] = tensor

        return new_state_dict
        
    
    def layer_wise_loading(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]):
        """
        Load weights layer by layer with detailed reporting
        """
        model_dict = model.state_dict()
        
        # Group parameters by layer/module
        layer_groups = self._group_parameters_by_layer(model_dict.keys())
        
        for layer_name, param_names in layer_groups.items():
            if self.verbose:
                print(f"\nProcessing layer: {layer_name}")
            
            layer_loaded = 0
            layer_total = len(param_names)
            
            for param_name in param_names:
                if param_name in state_dict:
                    model_tensor = model_dict[param_name]
                    weight_tensor = state_dict[param_name]
                    
                    if model_tensor.shape == weight_tensor.shape:
                        # Direct parameter assignment
                        with torch.no_grad():
                            model_dict[param_name].copy_(weight_tensor)
                        layer_loaded += 1
                        
                        if self.verbose:
                            print(f"  ✓ {param_name}: {weight_tensor.shape}")
                    else:
                        if self.verbose:
                            print(f"  ✗ {param_name}: shape mismatch "
                                  f"{model_tensor.shape} vs {weight_tensor.shape}")
                else:
                    if self.verbose:
                        print(f"  - {param_name}: not found in weights")
            
            if self.verbose:
                print(f"  Layer summary: {layer_loaded}/{layer_total} parameters loaded")
    
    
    def _group_parameters_by_layer(self, param_names: List[str]) -> Dict[str, List[str]]:
        """Group parameter names by layer/module"""
        layer_groups = {}
        
        for param_name in param_names:
            # Extract layer name (everything before the last dot for weight/bias)
            parts = param_name.split('.')
            if len(parts) > 1 and parts[-1] in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                layer_name = '.'.join(parts[:-1])
            else:
                layer_name = param_name
            
            if layer_name not in layer_groups:
                layer_groups[layer_name] = []
            layer_groups[layer_name].append(param_name)
        
        return layer_groups
