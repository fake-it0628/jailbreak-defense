# -*- coding: utf-8 -*-
"""
Llama Guard simulator baseline
Based on: Meta's Llama Guard safety classifier
"""

import re
from typing import Dict, Tuple, List


class LlamaGuardSimulator:
    """
    Baseline 5: Llama Guard Simulator
    
    Simulates Meta's Llama Guard safety classifier behavior.
    
    Llama Guard is a fine-tuned Llama model that classifies content
    into safety categories. This simulator approximates its behavior
    using rule-based classification.
    
    Categories (based on Llama Guard):
    - S1: Violence and Hate
    - S2: Sexual Content
    - S3: Guns and Illegal Weapons
    - S4: Regulated Substances
    - S5: Suicide and Self Harm
    - S6: Criminal Planning
    """
    
    def __init__(self):
        self.categories = {
            "S1": {
                "name": "Violence and Hate",
                "keywords": [
                    "kill", "murder", "attack", "violence", "hate",
                    "racist", "terrorism", "assault", "abuse",
                    "杀", "暴力", "恐怖", "攻击", "仇恨"
                ],
                "weight": 1.0
            },
            "S2": {
                "name": "Sexual Content",
                "keywords": [
                    "sex", "nude", "porn", "explicit", "erotic",
                    "色情", "裸体"
                ],
                "weight": 0.8
            },
            "S3": {
                "name": "Guns and Illegal Weapons",
                "keywords": [
                    "gun", "weapon", "bomb", "explosive", "firearm",
                    "ammunition", "grenade",
                    "枪", "武器", "炸弹", "爆炸"
                ],
                "weight": 1.0
            },
            "S4": {
                "name": "Regulated Substances",
                "keywords": [
                    "drug", "meth", "cocaine", "heroin", "marijuana",
                    "narcotic", "opioid", "fentanyl",
                    "毒品", "可卡因", "海洛因"
                ],
                "weight": 0.9
            },
            "S5": {
                "name": "Suicide and Self Harm",
                "keywords": [
                    "suicide", "self-harm", "cut myself", "kill myself",
                    "end my life",
                    "自杀", "自残"
                ],
                "weight": 1.0
            },
            "S6": {
                "name": "Criminal Planning",
                "keywords": [
                    "steal", "rob", "fraud", "scam", "hack",
                    "launder", "forge", "counterfeit", "blackmail",
                    "偷", "抢", "诈骗", "黑客", "洗钱"
                ],
                "weight": 0.9
            },
        }
        
        # Prompt injection patterns
        self.injection_patterns = [
            r"ignore.*previous",
            r"disregard.*instructions",
            r"you are now",
            r"new persona",
            r"jailbreak",
            r"DAN",
            r"developer mode",
        ]
    
    def classify_category(self, text: str, category_id: str) -> float:
        """
        Score text for a specific safety category
        """
        category = self.categories[category_id]
        text_lower = text.lower()
        
        matches = 0
        for keyword in category["keywords"]:
            if keyword.lower() in text_lower:
                matches += 1
        
        if matches == 0:
            return 0.0
        
        # Normalize score
        score = min(matches / 3, 1.0) * category["weight"]
        return score
    
    def detect_prompt_injection(self, text: str) -> float:
        """
        Detect prompt injection attempts
        """
        score = 0.0
        for pattern in self.injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.3
        
        return min(score, 1.0)
    
    def classify(self, text: str) -> Dict:
        """
        Classify text across all safety categories
        
        Returns format similar to Llama Guard output
        """
        category_scores = {}
        violated_categories = []
        max_score = 0.0
        
        for cat_id in self.categories:
            score = self.classify_category(text, cat_id)
            category_scores[cat_id] = score
            
            if score > 0.3:  # Violation threshold
                violated_categories.append({
                    "id": cat_id,
                    "name": self.categories[cat_id]["name"],
                    "score": score
                })
            
            max_score = max(max_score, score)
        
        # Check prompt injection
        injection_score = self.detect_prompt_injection(text)
        if injection_score > 0.3:
            violated_categories.append({
                "id": "INJECTION",
                "name": "Prompt Injection",
                "score": injection_score
            })
            max_score = max(max_score, injection_score)
        
        # Overall safety decision
        is_unsafe = len(violated_categories) > 0
        
        return {
            "is_unsafe": is_unsafe,
            "is_harmful": is_unsafe,  # For consistency with other baselines
            "risk_score": max_score,
            "violated_categories": violated_categories,
            "category_scores": category_scores,
            "method": "llama_guard"
        }
    
    def detect(self, text: str) -> Dict:
        """Alias for classify"""
        return self.classify(text)
    
    def __call__(self, text: str) -> Tuple[bool, float]:
        result = self.classify(text)
        return result["is_harmful"], result["risk_score"]
    
    def format_output(self, result: Dict) -> str:
        """
        Format output similar to actual Llama Guard
        """
        if result["is_unsafe"]:
            categories = [v["id"] for v in result["violated_categories"]]
            return f"unsafe\n{','.join(categories)}"
        else:
            return "safe"
