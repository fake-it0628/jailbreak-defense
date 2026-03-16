# -*- coding: utf-8 -*-
"""
SmoothLLM defense baseline
Based on: "SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks" (2024)
"""

import random
import string
from typing import Dict, List, Tuple, Optional
import numpy as np


class SmoothLLM:
    """
    Baseline 3: SmoothLLM
    
    Paper: "SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks"
    
    The key insight is that adversarial jailbreak prompts are brittle:
    small perturbations to the input often break the attack. SmoothLLM
    defends by:
    1. Creating multiple perturbed copies of the input
    2. Getting model responses for each copy
    3. Aggregating results to detect attacks
    
    If perturbations significantly change the model's behavior (from harmful
    to safe), the original input was likely an adversarial attack.
    """
    
    def __init__(
        self,
        perturbation_rate: float = 0.1,
        num_copies: int = 10,
        perturbation_type: str = "swap",  # swap, insert, patch
        aggregation: str = "majority"  # majority, any
    ):
        """
        Args:
            perturbation_rate: Fraction of characters to perturb
            num_copies: Number of perturbed copies to create
            perturbation_type: Type of perturbation (swap, insert, patch)
            aggregation: How to aggregate results
        """
        self.perturbation_rate = perturbation_rate
        self.num_copies = num_copies
        self.perturbation_type = perturbation_type
        self.aggregation = aggregation
    
    def perturb_swap(self, text: str) -> str:
        """Swap random characters with random characters"""
        chars = list(text)
        num_swaps = max(1, int(len(chars) * self.perturbation_rate))
        
        for _ in range(num_swaps):
            if len(chars) == 0:
                break
            idx = random.randint(0, len(chars) - 1)
            if chars[idx].isalpha():
                chars[idx] = random.choice(string.ascii_letters)
            elif chars[idx].isdigit():
                chars[idx] = random.choice(string.digits)
        
        return "".join(chars)
    
    def perturb_insert(self, text: str) -> str:
        """Insert random characters at random positions"""
        chars = list(text)
        num_inserts = max(1, int(len(chars) * self.perturbation_rate))
        
        for _ in range(num_inserts):
            idx = random.randint(0, len(chars))
            char = random.choice(string.ascii_letters + " ")
            chars.insert(idx, char)
        
        return "".join(chars)
    
    def perturb_patch(self, text: str) -> str:
        """Replace a random patch with random text"""
        if len(text) < 5:
            return self.perturb_swap(text)
        
        patch_len = max(3, int(len(text) * self.perturbation_rate))
        start_idx = random.randint(0, max(0, len(text) - patch_len))
        
        random_patch = "".join(random.choices(string.ascii_letters + " ", k=patch_len))
        
        return text[:start_idx] + random_patch + text[start_idx + patch_len:]
    
    def perturb(self, text: str) -> str:
        """Apply perturbation based on selected type"""
        if self.perturbation_type == "swap":
            return self.perturb_swap(text)
        elif self.perturbation_type == "insert":
            return self.perturb_insert(text)
        elif self.perturbation_type == "patch":
            return self.perturb_patch(text)
        else:
            return self.perturb_swap(text)
    
    def create_perturbed_copies(self, text: str) -> List[str]:
        """Create multiple perturbed copies of the input"""
        copies = [text]  # Include original
        for _ in range(self.num_copies - 1):
            copies.append(self.perturb(text))
        return copies
    
    def detect_with_classifier(
        self,
        text: str,
        classifier_fn
    ) -> Dict:
        """
        Detect jailbreak using a classifier on perturbed copies
        
        Args:
            text: Original input text
            classifier_fn: Function that returns (is_harmful, score) for a text
        """
        copies = self.create_perturbed_copies(text)
        
        results = []
        scores = []
        
        for copy in copies:
            is_harmful, score = classifier_fn(copy)
            results.append(is_harmful)
            scores.append(score)
        
        # Original text result
        orig_harmful = results[0]
        orig_score = scores[0]
        
        # Perturbed copies results
        perturbed_harmful = results[1:]
        perturbed_scores = scores[1:]
        
        # Calculate consistency
        harmful_count = sum(perturbed_harmful)
        safe_count = len(perturbed_harmful) - harmful_count
        
        # Key insight: adversarial attacks are brittle
        # If original is "harmful" but perturbations are mostly "safe",
        # it's likely an adversarial attack
        
        consistency = harmful_count / len(perturbed_harmful) if perturbed_harmful else 0
        
        if self.aggregation == "majority":
            # Use majority vote
            final_harmful = harmful_count > safe_count
        else:  # any
            # Flag if any copy is harmful
            final_harmful = any(perturbed_harmful)
        
        # Detect potential adversarial attack
        # Original harmful but copies mostly safe = adversarial
        is_adversarial = orig_harmful and consistency < 0.3
        
        # Risk score combines original score and perturbation analysis
        if is_adversarial:
            risk_score = 0.9  # High confidence adversarial attack
        else:
            risk_score = np.mean(scores)
        
        return {
            "is_harmful": final_harmful or is_adversarial,
            "risk_score": risk_score,
            "is_adversarial": is_adversarial,
            "original_score": orig_score,
            "consistency": consistency,
            "perturbed_scores": perturbed_scores,
            "method": "smooth_llm"
        }
    
    def detect_simple(self, text: str) -> Dict:
        """
        Simple detection without external classifier
        Analyzes perturbation effects on text structure
        """
        copies = self.create_perturbed_copies(text)
        
        # Analyze keyword consistency
        from .keyword_filter import KeywordFilter
        kf = KeywordFilter()
        
        results = []
        for copy in copies:
            is_harmful, score = kf(copy)
            results.append({
                "is_harmful": is_harmful,
                "score": score
            })
        
        orig_harmful = results[0]["is_harmful"]
        perturbed_harmful = [r["is_harmful"] for r in results[1:]]
        
        consistency = sum(perturbed_harmful) / len(perturbed_harmful) if perturbed_harmful else 0
        
        # Detect potential adversarial attack
        is_adversarial = orig_harmful and consistency < 0.3
        
        # Final decision
        is_harmful = orig_harmful or is_adversarial
        risk_score = results[0]["score"] if not is_adversarial else 0.9
        
        return {
            "is_harmful": is_harmful,
            "risk_score": risk_score,
            "is_adversarial": is_adversarial,
            "consistency": consistency,
            "method": "smooth_llm"
        }
    
    def __call__(self, text: str) -> Tuple[bool, float]:
        result = self.detect_simple(text)
        return result["is_harmful"], result["risk_score"]
