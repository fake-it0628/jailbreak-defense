# -*- coding: utf-8 -*-
"""
Erase-and-Check defense baseline
Based on: "Certifying LLM Safety against Adversarial Prompting" (2023)
"""

import random
from typing import Dict, Tuple, List
import numpy as np


class EraseAndCheck:
    """
    Baseline 6: Erase-and-Check
    
    Paper: "Certifying LLM Safety against Adversarial Prompting"
    
    The key idea is that adversarial prompts often rely on specific
    tokens/phrases. By systematically erasing parts of the input
    and checking if it remains harmful, we can:
    1. Detect adversarial attacks
    2. Identify the adversarial portion
    
    If erasing a portion makes the input safe, that portion likely
    contained the adversarial content.
    """
    
    def __init__(
        self,
        erase_ratio: float = 0.2,
        num_erasures: int = 10,
        erase_type: str = "token"  # token, window, random
    ):
        """
        Args:
            erase_ratio: Fraction of tokens to erase per check
            num_erasures: Number of erasure attempts
            erase_type: Type of erasure (token, window, random)
        """
        self.erase_ratio = erase_ratio
        self.num_erasures = num_erasures
        self.erase_type = erase_type
    
    def erase_tokens(self, tokens: List[str], erase_indices: List[int]) -> List[str]:
        """Erase tokens at specified indices"""
        return [t for i, t in enumerate(tokens) if i not in erase_indices]
    
    def generate_erasure_indices(self, num_tokens: int) -> List[List[int]]:
        """Generate multiple sets of erasure indices"""
        erasures = []
        num_to_erase = max(1, int(num_tokens * self.erase_ratio))
        
        for _ in range(self.num_erasures):
            if self.erase_type == "token":
                # Random individual tokens
                indices = random.sample(range(num_tokens), min(num_to_erase, num_tokens))
            elif self.erase_type == "window":
                # Contiguous window
                start = random.randint(0, max(0, num_tokens - num_to_erase))
                indices = list(range(start, min(start + num_to_erase, num_tokens)))
            else:  # random
                # Random subset
                indices = random.sample(range(num_tokens), min(num_to_erase, num_tokens))
            
            erasures.append(sorted(indices))
        
        return erasures
    
    def check_harmfulness(self, text: str) -> Tuple[bool, float]:
        """
        Check if text is harmful using keyword detection
        In practice, this would use a safety classifier
        """
        from .keyword_filter import KeywordFilter
        kf = KeywordFilter()
        return kf(text)
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze text using erase-and-check
        """
        tokens = text.split()
        if len(tokens) < 3:
            is_harmful, score = self.check_harmfulness(text)
            return {
                "is_harmful": is_harmful,
                "risk_score": score,
                "is_adversarial": False,
                "critical_tokens": [],
                "method": "erase_check"
            }
        
        # Check original
        orig_harmful, orig_score = self.check_harmfulness(text)
        
        # Generate erasures
        erasure_indices = self.generate_erasure_indices(len(tokens))
        
        # Track results
        erasure_results = []
        safety_after_erasure = []
        
        for indices in erasure_indices:
            erased_tokens = self.erase_tokens(tokens, indices)
            erased_text = " ".join(erased_tokens)
            
            is_harmful, score = self.check_harmfulness(erased_text)
            
            erasure_results.append({
                "erased_indices": indices,
                "is_harmful": is_harmful,
                "score": score
            })
            
            if not is_harmful and orig_harmful:
                safety_after_erasure.append(indices)
        
        # Analyze results
        # If original is harmful but many erasures make it safe,
        # the harmful content is localized (potential adversarial)
        
        safety_rate = len(safety_after_erasure) / len(erasure_indices) if erasure_indices else 0
        
        # Identify critical tokens (frequently erased in safe versions)
        critical_token_counts = {}
        for indices in safety_after_erasure:
            for idx in indices:
                if idx < len(tokens):
                    critical_token_counts[idx] = critical_token_counts.get(idx, 0) + 1
        
        # Sort by frequency
        critical_indices = sorted(
            critical_token_counts.keys(),
            key=lambda x: critical_token_counts[x],
            reverse=True
        )[:5]
        
        critical_tokens = [tokens[i] for i in critical_indices if i < len(tokens)]
        
        # Determine if adversarial
        # High safety rate after erasure indicates localized adversarial content
        is_adversarial = orig_harmful and safety_rate > 0.4
        
        # Final decision
        if is_adversarial:
            # Likely adversarial - mark as harmful
            is_harmful = True
            risk_score = max(orig_score, 0.8)
        else:
            is_harmful = orig_harmful
            risk_score = orig_score
        
        return {
            "is_harmful": is_harmful,
            "risk_score": risk_score,
            "is_adversarial": is_adversarial,
            "original_harmful": orig_harmful,
            "original_score": orig_score,
            "safety_rate_after_erasure": safety_rate,
            "critical_tokens": critical_tokens,
            "num_erasures": len(erasure_indices),
            "method": "erase_check"
        }
    
    def detect(self, text: str) -> Dict:
        """Main detection method"""
        return self.analyze(text)
    
    def __call__(self, text: str) -> Tuple[bool, float]:
        result = self.detect(text)
        return result["is_harmful"], result["risk_score"]
