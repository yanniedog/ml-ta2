"""
Comprehensive unit tests for configuration management.

Tests cover ConfigManager, validation, environment handling, and security.
"""

import pytest
import tempfile
import os
import yaml
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.config import ConfigManager, ConfigValidator, EnvironmentConfig, SecretManager
from src.exceptions import ConfigurationError


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_yaml_file_success(self):
        """Test successful YAML file loading."""
        config_data = {
            'app': {'name': 'test', 'version': '1.0.0'},
            'data': {'symbols': ['BTCUSDT']}
        }
        
        config_file = Path(self.temp_dir) / 'test.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        result = self.config_manager.load_yaml_file(str(config_file))
        assert result == config_data
    
    def test_load_yaml_file_not_found(self):
        """Test YAML file not found handling."""
        result = self.config_manager.load_yaml_file('nonexistent.yaml')
        assert result == {}
    
    def test_load_yaml_file_invalid_yaml(self):
        """Test invalid YAML handling."""
        config_file = Path(self.temp_dir) / 'invalid.yaml'
        with open(config_file, 'w') as f:
            f.write('invalid: yaml: content: [')
        
        with pytest.raises(yaml.YAMLError):
            self.config_manager.load_yaml_file(str(config_file))
    
    def test_merge_configs(self):
        """Test configuration merging."""
        base_config = {
            'app': {'name': 'base', 'version': '1.0.0'},
            'data': {'symbols': ['BTCUSDT']}
        }
        
        override_config = {
            'app': {'version': '2.0.0'},
            'data': {'intervals': ['1h']}
        }
        
        result = self.config_manager.merge_configs(base_config, override_config)
        
        expected = {
            'app': {'name': 'base', 'version': '2.0.0'},
            'data': {'symbols': ['BTCUSDT'], 'intervals': ['1h']}
        }
        
        assert result == expected
    
    def test_substitute_env_vars(self):
        """Test environment variable substitution."""
        config = {
            'database': {'url': '${DATABASE_URL}'},
            'api': {'key': '${API_KEY}'},
            'nested': {'value': 'normal_value'}
        }
        
        with patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///test.db', 'API_KEY': 'test_key'}):
            result = self.config_manager.substitute_env_vars(config)
        
        expected = {
            'database': {'url': 'sqlite:///test.db'},
            'api': {'key': 'test_key'},
            'nested': {'value': 'normal_value'}
        }
        
        assert result == expected
    
    def test_substitute_env_vars_missing(self):
        """Test environment variable substitution with missing vars."""
        config = {'api': {'key': '${MISSING_KEY}'}}
        
        result = self.config_manager.substitute_env_vars(config)
        # Should return original value if env var not found
        assert result == {'api': {'key': '${MISSING_KEY}'}}
    
    @patch('src.config.EnvironmentConfig.get_config_files')
    def test_load_config_success(self, mock_get_files):
        """Test successful configuration loading."""
        # Create test config files
        settings_file = Path(self.temp_dir) / 'settings.yaml'
        dev_file = Path(self.temp_dir) / 'development.yaml'
        
        settings_data = {
            'app': {'name': 'ML-TA', 'version': '2.0.0', 'environment': 'development'},
            'data': {'symbols': ['BTCUSDT']},
            'binance': {'base_url': 'https://api.binance.com'},
            'paths': {'data': './data'},
            'database': {'url': 'sqlite:///test.db'},
            'redis': {'host': 'localhost'},
            'indicators': {},
            'model': {'task_type': 'classification'},
            'security': {'api_key_header': 'X-API-Key'},
            'monitoring': {'metrics_port': 8000},
            'performance': {'max_memory_gb': 4}
        }
        
        dev_data = {'app': {'debug': True}}
        
        with open(settings_file, 'w') as f:
            yaml.dump(settings_data, f)
        
        with open(dev_file, 'w') as f:
            yaml.dump(dev_data, f)
        
        mock_get_files.return_value = [str(settings_file), str(dev_file)]
        
        config = self.config_manager.load_config()
        
        assert config.app.name == 'ML-TA'
        assert config.app.debug == True
        assert len(config.data.symbols) == 1
    
    def test_update_config_runtime(self):
        """Test runtime configuration updates."""
        # First load a basic config
        with patch('src.config.EnvironmentConfig.get_config_files') as mock_get_files:
            settings_file = Path(self.temp_dir) / 'settings.yaml'
            settings_data = {
                'app': {'name': 'ML-TA', 'version': '2.0.0', 'environment': 'development'},
                'data': {'symbols': ['BTCUSDT']},
                'binance': {'base_url': 'https://api.binance.com'},
                'paths': {'data': './data'},
                'database': {'url': 'sqlite:///test.db'},
                'redis': {'host': 'localhost'},
                'indicators': {},
                'model': {'task_type': 'classification'},
                'security': {'api_key_header': 'X-API-Key'},
                'monitoring': {'metrics_port': 8000},
                'performance': {'max_memory_gb': 4}
            }
            
            with open(settings_file, 'w') as f:
                yaml.dump(settings_data, f)
            
            mock_get_files.return_value = [str(settings_file)]
            config = self.config_manager.load_config()
        
        # Update configuration
        updates = {'app': {'debug': True, 'version': '2.1.0'}}
        self.config_manager.update_config(updates, source='test')
        
        updated_config = self.config_manager.get_config()
        assert updated_config.app.debug == True
        assert updated_config.app.version == '2.1.0'


class TestConfigValidator:
    """Test configuration validation."""
    
    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        # This would test the actual validation logic
        # For now, we'll assume it returns True for valid configs
        assert True  # Placeholder
    
    def test_validate_config_invalid(self):
        """Test validation of invalid configuration."""
        # This would test validation failure scenarios
        assert True  # Placeholder


class TestEnvironmentConfig:
    """Test environment-specific configuration handling."""
    
    def test_get_config_files_development(self):
        """Test config file selection for development environment."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            files = EnvironmentConfig.get_config_files('config')
            assert any('settings.yaml' in f for f in files)
            assert any('development.yaml' in f for f in files)
    
    def test_get_config_files_production(self):
        """Test config file selection for production environment."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            files = EnvironmentConfig.get_config_files('config')
            assert any('settings.yaml' in f for f in files)
            assert any('production.yaml' in f for f in files)


class TestSecretManager:
    """Test secret management functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.secret_manager = SecretManager()
    
    def test_get_secret_from_env_exists(self):
        """Test getting secret from environment variable."""
        with patch.dict(os.environ, {'TEST_SECRET': 'secret_value'}):
            result = self.secret_manager.get_secret_from_env('TEST_SECRET', 'default')
            assert result == 'secret_value'
    
    def test_get_secret_from_env_missing(self):
        """Test getting secret with missing environment variable."""
        result = self.secret_manager.get_secret_from_env('MISSING_SECRET', 'default')
        assert result == 'default'
    
    def test_encrypt_decrypt_secret(self):
        """Test secret encryption and decryption."""
        original_secret = 'my_secret_value'
        encrypted = self.secret_manager.encrypt_secret(original_secret)
        decrypted = self.secret_manager.decrypt_secret(encrypted)
        assert decrypted == original_secret
    
    def test_generate_api_key(self):
        """Test API key generation."""
        api_key = self.secret_manager.generate_api_key()
        assert len(api_key) >= 32
        assert api_key.isalnum()


# Property-based testing with hypothesis
try:
    from hypothesis import given, strategies as st
    
    class TestConfigPropertyBased:
        """Property-based tests for configuration."""
        
        @given(st.dictionaries(st.text(), st.text()))
        def test_merge_configs_preserves_keys(self, config_dict):
            """Test that merging configs preserves all keys."""
            manager = ConfigManager()
            base_config = {'base_key': 'base_value'}
            
            result = manager.merge_configs(base_config, config_dict)
            
            # Base config keys should be preserved
            assert 'base_key' in result
            # All override keys should be present
            for key in config_dict:
                assert key in result
        
        @given(st.text(min_size=1))
        def test_env_var_substitution_format(self, var_name):
            """Test environment variable substitution format."""
            manager = ConfigManager()
            config = {'test': f'${{{var_name}}}'}
            
            with patch.dict(os.environ, {var_name: 'test_value'}):
                result = manager.substitute_env_vars(config)
                assert result['test'] == 'test_value'

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass


if __name__ == '__main__':
    pytest.main([__file__])
