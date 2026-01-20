"""
Multilingual Speech Recognition using OpenAI Whisper
Supports 100+ languages with automatic detection
"""

import whisper
import torch
from typing import Dict, Optional
import os
from pathlib import Path


class MultilingualSpeechRecognizer:
    """
    Speech recognition with automatic language detection
    Based on OpenAI Whisper (2024)
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model
        
        Args:
            model_size: 'tiny', 'base', 'small', 'medium', 'large'
                       (base is good balance of speed/accuracy)
        """
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"✓ Whisper loaded on {self.device}")
        
        # Language name mapping
        self.language_names = {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'mr': 'Marathi',
            'gu': 'Gujarati',
            'bn': 'Bengali',
            'pa': 'Punjabi',
            'ur': 'Urdu',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic'
        }
    
    def transcribe_voice(self, audio_file_path: str, task: str = 'transcribe') -> Dict:
        """
        Transcribe voice input with automatic language detection
        
        Args:
            audio_file_path: Path to audio file (wav, mp3, etc.)
            task: 'transcribe' (keep original language) or 'translate' (to English)
        
        Returns:
            {
                'text': 'transcribed text',
                'language': 'detected language code (e.g., hi, en)',
                'language_name': 'Hindi/English/etc',
                'confidence': 0.95,
                'segments': [...] (detailed breakdown)
            }
        """
        try:
            # Transcribe with language detection
            result = self.model.transcribe(
                audio_file_path,
                task=task,
                language=None,  # Auto-detect
                fp16=False,
                verbose=False
            )
            
            detected_lang = result.get('language', 'en')
            
            return {
                'text': result['text'].strip(),
                'language': detected_lang,
                'language_name': self.language_names.get(detected_lang, detected_lang.upper()),
                'confidence': result.get('language_probability', 0.0),
                'segments': result.get('segments', []),
                'duration': sum(seg['end'] - seg['start'] for seg in result.get('segments', []))
            }
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return {
                'text': '',
                'language': 'en',
                'language_name': 'English',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def extract_poster_info(self, transcription: str, language: str) -> Dict:
        """
        Extract structured information from transcription
        Uses simple keyword extraction (can be enhanced with LLM)
        
        Returns:
            {
                'product_name': '...',
                'visual_theme': '...',
                'business_name': '...',
                'mentioned_discount': '...' or None
            }
        """
        # Simple extraction (in production, use LLM for better results)
        text_lower = transcription.lower()
        
        # Extract product name (simple heuristic)
        product_name = self._extract_product_name(transcription, language)
        
        # Extract visual theme
        visual_theme = self._extract_visual_theme(text_lower)
        
        # Extract mentioned discount
        mentioned_discount = self._extract_discount(text_lower)
        
        # Extract business name (if mentioned)
        business_name = self._extract_business_name(transcription)
        
        return {
            'product_name': product_name,
            'visual_theme': visual_theme,
            'business_name': business_name,
            'mentioned_discount': mentioned_discount,
            'full_transcription': transcription
        }
    
    def _extract_product_name(self, text: str, language: str) -> str:
        """Extract product name from transcription"""
        # Simple: use first few words or full text if short
        words = text.split()
        if len(words) <= 5:
            return text
        return ' '.join(words[:5])
    
    def _extract_visual_theme(self, text_lower: str) -> str:
        """Extract visual theme keywords"""
        themes = {
            'vibrant': ['vibrant', 'colorful', 'bright', 'रंगीन'],
            'fresh': ['fresh', 'organic', 'natural', 'ताज़ा'],
            'professional': ['professional', 'elegant', 'premium'],
            'festive': ['festival', 'celebration', 'festive', 'त्योहार'],
            'modern': ['modern', 'contemporary', 'sleek']
        }
        
        for theme, keywords in themes.items():
            if any(kw in text_lower for kw in keywords):
                return theme
        
        return 'vibrant'  # Default
    
    def _extract_discount(self, text_lower: str) -> Optional[str]:
        """Extract mentioned discount percentage"""
        import re
        
        # Look for patterns like "20%", "20 percent", "discount"
        patterns = [
            r'(\d+)\s*%',
            r'(\d+)\s*percent',
            r'discount.*?(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return f"{match.group(1)}%"
        
        return None
    
    def _extract_business_name(self, text: str) -> str:
        """Extract business name (simple heuristic)"""
        # In production, use NER or LLM
        # For now, return empty (will be filled from user profile)
        return ""


# Singleton instance
speech_recognizer = MultilingualSpeechRecognizer(model_size="base")
