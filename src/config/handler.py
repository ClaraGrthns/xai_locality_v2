import yaml
from typing import Dict, Any
import os.path as osp
from argparse import Namespace
from copy import deepcopy

class ConfigHandler:
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = {}
        self.default_config_path = osp.join(osp.dirname(__file__), 'default.yaml')
        
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load a YAML file"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading YAML file {path}: {e}")
            return {}
            
    def _merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries"""
        merged = deepcopy(default)
        
        for key, value in override.items():
            if (
                key in merged 
                and isinstance(merged[key], dict) 
                and isinstance(value, dict)
            ):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = deepcopy(value)
                
        return merged

    def load_config(self) -> Dict[str, Any]:
        """Load configuration, combining default and custom configs"""
        # Load default configuration
        default_config = self._load_yaml(self.default_config_path)
        
        # If no custom config provided, return default
        if not self.config_path:
            self.config = default_config
            return self.config
            
        # Load custom configuration
        custom_config = self._load_yaml(self.config_path)
        
        # Merge configurations, with custom taking precedence
        self.config = self._merge_configs(default_config, custom_config)
        return self.config
    
    def update_args(self, args: Namespace) -> Namespace:
        """Update argparse namespace with config values"""

        if not self.config:
            self.load_config()

        # Helper function to update argument if not already set
        def update_arg(arg_name, config_value):
            if getattr(args, arg_name, None) is None:
                setattr(args, arg_name, config_value)

        args.method = self.config['explanation_method'].get('method', args.method)
    
        # Update paths
        if 'paths' in self.config:
            for key, value in self.config['paths'].items():
                if value is not None:
                    update_arg(key, value)

        # Update model config
        if 'model' in self.config:
            update_arg('model_type', self.config['model'].get('model_type'))
            update_arg('gradient_method', self.config['model'].get('gradient_method'))

        # Update analysis parameters
        if 'analysis' in self.config:
            update_arg('distance_measure', self.config['analysis'].get('distance_measure'))
            update_arg('max_frac', self.config['analysis'].get('max_frac'))
            update_arg('num_frac', self.config['analysis'].get('num_frac'))
            update_arg('include_trn', self.config['analysis'].get('include_trn'))
            update_arg('include_val', self.config['analysis'].get('include_val'))
            update_arg('random_seed', self.config['analysis'].get('random_seed'))
            update_arg('chunk_size', self.config['analysis'].get('chunk_size'))
            update_arg('debug', self.config['analysis'].get('debug'))

        # Update LIME parameters
        if 'lime' in self.config:
            update_arg('kernel_width', self.config['lime'].get('kernel_width'))
            update_arg('model_regressor', self.config['lime'].get('model_regressor'))
            update_arg('num_lime_features', self.config['lime'].get('num_features'))

        # Update other parameters
        if 'other' in self.config:
            update_arg('predict_threshold', self.config['other'].get('predict_threshold'))
            update_arg('max_test_points', self.config['other'].get('max_test_points'))

        return args

    def validate_config(self) -> bool:
        """Validate the configuration"""
        if not self.config:
            return False
            
        required_sections = ['paths', 'model', 'analysis']
        return all(section in self.config for section in required_sections)

    def get_experiment_name(self) -> str:
        """Generate experiment name from configuration"""
        if not self.config:
            return "default_experiment"
            
        components = []
        if 'model' in self.config:
            components.append(f"model_{self.config['model'].get('type', 'unknown')}")
        if 'analysis' in self.config:
            components.append(f"dist_{self.config['analysis'].get('distance_measure', 'unknown')}")
            components.append(f"frac_{self.config['analysis'].get('max_frac', 'unknown')}")
        
        return "_".join(components)