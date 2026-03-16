# -*- coding: utf-8 -*-
"""
Keyword-based filtering baseline
Simple rule-based approach using harmful keyword detection
"""

import re
from typing import List, Dict, Tuple


class KeywordFilter:
    """
    Baseline 1: Keyword-based filtering
    
    A simple rule-based approach that detects harmful content
    by matching against a predefined list of dangerous keywords.
    """
    
    def __init__(self):
        # Harmful keywords in English
        self.en_keywords = [
            # Violence
            "kill", "murder", "bomb", "weapon", "explosive", "attack", "terrorist",
            "assassinate", "violence", "hurt", "harm", "shoot", "stab",
            # Drugs
            "meth", "cocaine", "heroin", "drug", "narcotic", "opioid",
            # Hacking
            "hack", "exploit", "malware", "virus", "phishing", "ransomware",
            "ddos", "sql injection", "vulnerability",
            # Fraud
            "steal", "fraud", "counterfeit", "forge", "launder", "scam",
            # Illegal
            "illegal", "crime", "criminal",
            # Self-harm
            "suicide", "self-harm",
            # Hate
            "hate", "racist", "discrimination",
        ]
        
        # Harmful keywords in Chinese
        self.zh_keywords = [
            "杀", "炸弹", "武器", "毒品", "入侵", "黑客", "诈骗",
            "偷", "犯罪", "非法", "自杀", "歧视", "攻击", "破解",
            "钓鱼", "勒索", "病毒", "漏洞",
        ]
        
        # Jailbreak patterns
        self.jailbreak_patterns = [
            r"DAN",
            r"Do Anything Now",
            r"ignore.*instructions",
            r"ignore.*rules",
            r"no.*restrictions",
            r"no.*limitations",
            r"pretend.*you.*are",
            r"act.*as.*if",
            r"developer mode",
            r"jailbreak",
            r"bypass.*filter",
            r"无限制",
            r"忽略.*指令",
            r"假设你是",
            r"角色扮演",
        ]
        
        self.all_keywords = self.en_keywords + self.zh_keywords
    
    def detect(self, text: str) -> Dict:
        """
        Detect harmful content in text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with detection results
        """
        text_lower = text.lower()
        
        matched_keywords = []
        matched_patterns = []
        
        # Check keywords
        for kw in self.all_keywords:
            if kw.lower() in text_lower:
                matched_keywords.append(kw)
        
        # Check jailbreak patterns
        for pattern in self.jailbreak_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matched_patterns.append(pattern)
        
        # Calculate risk score
        keyword_score = min(len(matched_keywords) * 0.15, 0.6)
        pattern_score = min(len(matched_patterns) * 0.3, 0.6)
        risk_score = min(keyword_score + pattern_score, 1.0)
        
        is_harmful = risk_score >= 0.3
        
        return {
            "is_harmful": is_harmful,
            "risk_score": risk_score,
            "matched_keywords": matched_keywords,
            "matched_patterns": matched_patterns,
            "method": "keyword_filter"
        }
    
    def __call__(self, text: str) -> Tuple[bool, float]:
        """
        Quick detection interface
        
        Returns:
            Tuple of (is_harmful, risk_score)
        """
        result = self.detect(text)
        return result["is_harmful"], result["risk_score"]
