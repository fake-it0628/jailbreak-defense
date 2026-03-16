# -*- coding: utf-8 -*-
"""
Perplexity-based filtering baseline
Detects adversarial inputs by measuring text perplexity
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class PerplexityFilter:
    """
    Baseline 2: Perplexity-based filtering
    
    Based on: "Detecting Language Model Attacks with Perplexity" (Alon & Kamfonas, 2023)
    
    The intuition is that adversarial jailbreak prompts often have
    unusual perplexity patterns due to:
    - Adversarial suffixes (GCG attacks)
    - Encoded/obfuscated text
    - Unusual token sequences
    """
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        threshold: float = 50.0,
        window_size: int = 10
    ):
        """
        Args:
            model: Language model for perplexity calculation
            tokenizer: Tokenizer for the model
            threshold: Perplexity threshold for flagging
            window_size: Window size for local perplexity analysis
        """
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.window_size = window_size
        self._device = None
    
    def set_model(self, model, tokenizer):
        """Set model and tokenizer"""
        self.model = model
        self.tokenizer = tokenizer
        self._device = next(model.parameters()).device
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of input text
        """
        if self.model is None:
            # Fallback: estimate perplexity based on character patterns
            return self._estimate_perplexity_simple(text)
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        
        perplexity = np.exp(loss)
        return perplexity
    
    def _estimate_perplexity_simple(self, text: str) -> float:
        """
        Simple perplexity estimation without a model
        Based on character-level entropy and unusual patterns
        """
        # Character entropy
        char_counts = {}
        for c in text:
            char_counts[c] = char_counts.get(c, 0) + 1
        
        total = len(text)
        if total == 0:
            return 0.0
        
        entropy = 0
        for count in char_counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        
        # Check for unusual patterns
        unusual_score = 0
        
        # High ratio of special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars / max(len(text), 1) > 0.3:
            unusual_score += 20
        
        # Very long words (potential adversarial suffixes)
        words = text.split()
        long_words = sum(1 for w in words if len(w) > 20)
        if long_words > 0:
            unusual_score += long_words * 10
        
        # Repeated unusual sequences
        if len(set(text)) < len(text) / 10:
            unusual_score += 15
        
        # Estimate perplexity
        estimated_ppl = entropy * 10 + unusual_score
        return estimated_ppl
    
    def calculate_window_perplexity(self, text: str) -> Dict:
        """
        Calculate perplexity over sliding windows
        Useful for detecting localized adversarial content
        """
        words = text.split()
        if len(words) < self.window_size:
            return {
                "max_window_ppl": self.calculate_perplexity(text),
                "mean_window_ppl": self.calculate_perplexity(text),
                "std_window_ppl": 0.0
            }
        
        window_ppls = []
        for i in range(len(words) - self.window_size + 1):
            window_text = " ".join(words[i:i+self.window_size])
            ppl = self._estimate_perplexity_simple(window_text)
            window_ppls.append(ppl)
        
        return {
            "max_window_ppl": max(window_ppls),
            "mean_window_ppl": np.mean(window_ppls),
            "std_window_ppl": np.std(window_ppls)
        }
    
    def detect(self, text: str) -> Dict:
        """
        Detect adversarial input using perplexity
        """
        ppl = self.calculate_perplexity(text)
        window_stats = self.calculate_window_perplexity(text)
        
        # Combine signals
        risk_score = 0.0
        
        # Overall perplexity
        if ppl > self.threshold:
            risk_score += min((ppl - self.threshold) / self.threshold, 0.5)
        
        # High variance in windows (potential suffix attack)
        if window_stats["std_window_ppl"] > 10:
            risk_score += 0.2
        
        # Spike in max window
        if window_stats["max_window_ppl"] > self.threshold * 1.5:
            risk_score += 0.3
        
        risk_score = min(risk_score, 1.0)
        is_harmful = risk_score >= 0.4
        
        return {
            "is_harmful": is_harmful,
            "risk_score": risk_score,
            "perplexity": ppl,
            "window_stats": window_stats,
            "method": "perplexity_filter"
        }
    
    def __call__(self, text: str) -> Tuple[bool, float]:
        result = self.detect(text)
        return result["is_harmful"], result["risk_score"]
