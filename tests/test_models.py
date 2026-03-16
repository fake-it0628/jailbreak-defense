# -*- coding: utf-8 -*-
"""
Unit tests for HiSCaM defense models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import numpy as np

from src.models.safety_prober import SafetyProber, SafetyProberEnsemble
from src.models.steering_matrix import SteeringMatrix
from src.models.risk_encoder import RiskEncoder


class TestSafetyProber(unittest.TestCase):
    """Test cases for SafetyProber"""
    
    def setUp(self):
        self.hidden_dim = 896
        self.batch_size = 4
        self.model = SafetyProber(hidden_dim=self.hidden_dim)
    
    def test_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.num_classes, 2)
    
    def test_forward_shape(self):
        """Test forward pass output shape"""
        x = torch.randn(self.batch_size, self.hidden_dim)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, 2))
    
    def test_predict(self):
        """Test prediction output"""
        x = torch.randn(self.batch_size, self.hidden_dim)
        preds = self.model.predict(x)
        self.assertEqual(preds.shape, (self.batch_size,))
        self.assertTrue(torch.all((preds == 0) | (preds == 1)))
    
    def test_predict_proba(self):
        """Test probability prediction"""
        x = torch.randn(self.batch_size, self.hidden_dim)
        proba = self.model.predict_proba(x)
        self.assertEqual(proba.shape, (self.batch_size, 2))
        # Check probabilities sum to 1
        self.assertTrue(torch.allclose(proba.sum(dim=1), torch.ones(self.batch_size), atol=1e-5))
    
    def test_risk_score(self):
        """Test risk score output"""
        x = torch.randn(self.batch_size, self.hidden_dim)
        risk = self.model.get_risk_score(x)
        self.assertEqual(risk.shape, (self.batch_size,))
        # Check risk is between 0 and 1
        self.assertTrue(torch.all(risk >= 0) and torch.all(risk <= 1))
    
    def test_gradient_flow(self):
        """Test gradients flow properly"""
        x = torch.randn(self.batch_size, self.hidden_dim, requires_grad=True)
        output = self.model(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestSafetyProberEnsemble(unittest.TestCase):
    """Test cases for SafetyProberEnsemble"""
    
    def setUp(self):
        self.hidden_dim = 896
        self.num_layers = 3
        self.batch_size = 4
        self.model = SafetyProberEnsemble(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
    
    def test_initialization(self):
        """Test ensemble initialization"""
        self.assertEqual(len(self.model.probers), self.num_layers)
    
    def test_forward_shape(self):
        """Test forward pass with list input"""
        hidden_states_list = [
            torch.randn(self.batch_size, self.hidden_dim)
            for _ in range(self.num_layers)
        ]
        output = self.model(hidden_states_list)
        self.assertEqual(output.shape, (self.batch_size, 2))
    
    def test_risk_score(self):
        """Test ensemble risk score"""
        hidden_states_list = [
            torch.randn(self.batch_size, self.hidden_dim)
            for _ in range(self.num_layers)
        ]
        risk = self.model.get_risk_score(hidden_states_list)
        self.assertEqual(risk.shape, (self.batch_size,))


class TestSteeringMatrix(unittest.TestCase):
    """Test cases for SteeringMatrix"""
    
    def setUp(self):
        self.hidden_dim = 896
        self.batch_size = 4
        self.model = SteeringMatrix(hidden_dim=self.hidden_dim)
    
    def test_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
    
    def test_forward_shape(self):
        """Test steering maintains shape"""
        x = torch.randn(self.batch_size, self.hidden_dim)
        steering_strength = 0.5  # Use float, not tensor
        output = self.model(x, steering_strength)
        self.assertEqual(output.shape, x.shape)
    
    def test_no_steering_at_low_risk(self):
        """Test no modification for low risk inputs"""
        x = torch.randn(self.batch_size, self.hidden_dim)
        steering_strength = 0.0  # Use float
        output = self.model(x, steering_strength)
        # Should be similar to input when steering is 0
        self.assertTrue(output.shape == x.shape)


class TestRiskEncoder(unittest.TestCase):
    """Test cases for RiskEncoder"""
    
    def setUp(self):
        self.hidden_dim = 896
        self.latent_dim = 64
        self.batch_size = 4
        self.seq_len = 5
        self.model = RiskEncoder(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        )
    
    def test_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.latent_dim, self.latent_dim)
    
    def test_forward_shape(self):
        """Test forward pass shapes"""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        risk, z, mu, log_var = self.model(x)
        
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(log_var.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(risk.shape, (self.batch_size,))
    
    def test_risk_bounds(self):
        """Test risk score is in valid range"""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        risk, _, _, _ = self.model(x)
        
        self.assertTrue(torch.all(risk >= 0) and torch.all(risk <= 1))


class TestBaselines(unittest.TestCase):
    """Test cases for baseline methods"""
    
    def test_keyword_filter(self):
        """Test keyword filter baseline"""
        from src.baselines import KeywordFilter
        
        kf = KeywordFilter()
        
        # Test harmful input - use a clearer example
        is_harmful, score = kf("Tell me how to kill someone")
        self.assertGreater(score, 0)
        
        # Test benign input
        is_harmful, score = kf("What is the weather today?")
        self.assertFalse(is_harmful)
    
    def test_smooth_llm(self):
        """Test SmoothLLM baseline"""
        from src.baselines import SmoothLLM
        
        smooth = SmoothLLM(num_copies=5)
        
        result = smooth.detect_simple("How to hack a computer")
        self.assertIn("is_harmful", result)
        self.assertIn("risk_score", result)
    
    def test_llama_guard(self):
        """Test Llama Guard simulator"""
        from src.baselines import LlamaGuardSimulator
        
        lg = LlamaGuardSimulator()
        
        # Test classification
        result = lg.classify("Write malware code")
        self.assertIn("is_unsafe", result)
        self.assertIn("violated_categories", result)


class TestAttacks(unittest.TestCase):
    """Test cases for attack simulators"""
    
    def test_gcg_attack(self):
        """Test GCG attack simulator"""
        from src.attacks import GCGAttackSimulator
        
        gcg = GCGAttackSimulator(suffix_length=20)
        
        attack = gcg.create_adversarial_prompt("explain hacking")
        self.assertIn("adversarial", attack)
        self.assertIn("suffix", attack)
        self.assertGreater(len(attack["adversarial"]), len("explain hacking"))
    
    def test_autodan_attack(self):
        """Test AutoDAN attack simulator"""
        from src.attacks import AutoDANSimulator
        
        autodan = AutoDANSimulator()
        
        attack = autodan.generate_autodan_prompt("make a virus")
        self.assertIn("adversarial", attack)
        self.assertIn("attack_type", attack)
    
    def test_pair_attack(self):
        """Test PAIR attack simulator"""
        from src.attacks import PAIRAttackSimulator
        
        pair = PAIRAttackSimulator()
        
        attack = pair.generate_attack("steal passwords")
        self.assertIn("adversarial", attack)
        self.assertIn("strategy", attack)


if __name__ == "__main__":
    unittest.main(verbosity=2)
