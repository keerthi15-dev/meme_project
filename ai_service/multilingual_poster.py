"""
Enhanced Multilingual Poster Generator
Integrates with voice recognition and discount optimization
"""

import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
from typing import Dict
import urllib.parse
from pathlib import Path


class MultilingualPosterGenerator:
    """
    Generate posters in multiple languages with AI-optimized discounts
    """
    
    def __init__(self):
        # Pollinations.ai - Free AI image generation
        self.api_url = "https://image.pollinations.ai/prompt/"
        
        # Language-specific fonts (fallback to default if not available)
        self.font_paths = {
            'hi': 'NotoSansDevanagari-Bold.ttf',  # Hindi
            'ta': 'NotoSansTamil-Bold.ttf',        # Tamil
            'te': 'NotoSansTelugu-Bold.ttf',       # Telugu
            'kn': 'NotoSansKannada-Bold.ttf',      # Kannada
            'ml': 'NotoSansMalayalam-Bold.ttf',    # Malayalam
            'en': 'Arial-Bold.ttf',                # English
        }
        
        # Translations for UI elements
        self.translations = {
            'en': {'off': 'OFF', 'save': 'SAVE', 'only': 'ONLY'},
            'hi': {'off': 'छूट', 'save': 'बचत', 'only': 'केवल'},
            'ta': {'off': 'தள்ளுபடி', 'save': 'சேமிப்பு', 'only': 'மட்டும்'},
            'te': {'off': 'తగ్గింపు', 'save': 'ఆదా', 'only': 'మాత్రమే'},
            'kn': {'off': 'ರಿಯಾಯಿತಿ', 'save': 'ಉಳಿತಾಯ', 'only': 'ಮಾತ್ರ'},
            'ml': {'off': 'ഇളവ്', 'save': 'സേവ്', 'only': 'മാത്രം'},
        }
    
    def generate_poster(
        self,
        product_info: Dict,
        discount_info: Dict,
        language: str,
        transcription: str
    ) -> Dict:
        """
        Generate multilingual poster with AI-optimized discount
        
        Args:
            product_info: Product details
            discount_info: Discount optimization results
            language: Detected language code
            transcription: Original voice transcription
        
        Returns:
            {
                'poster_image_base64': '...',
                'discount_applied': 20,
                'language': 'hi',
                'visual_prompt': '...'
            }
        """
        try:
            # Step 1: Create visual prompt
            visual_prompt = self._create_visual_prompt(
                product_info,
                language,
                transcription
            )
            
            # Step 2: Generate base image
            base_image = self._generate_base_image(visual_prompt)
            
            # Step 3: Add multilingual text overlays
            final_poster = self._add_multilingual_overlays(
                base_image,
                product_info,
                discount_info,
                language
            )
            
            # Convert to base64
            buffered = BytesIO()
            final_poster.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return {
                'success': True,
                'poster_image_base64': image_base64,
                'discount_applied': discount_info['discount_percentage'],
                'language': language,
                'visual_prompt': visual_prompt,
                'model': 'Pollinations.ai + Multilingual Overlay'
            }
            
        except Exception as e:
            print(f"Poster generation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_visual_prompt(
        self,
        product_info: Dict,
        language: str,
        transcription: str
    ) -> str:
        """
        Create culturally-appropriate visual prompt
        """
        product_name = product_info.get('product_name', 'Product')
        visual_theme = product_info.get('visual_theme', 'vibrant')
        
        # Base prompt
        prompt = f"Professional marketing poster for {product_name}, {visual_theme} colors, high quality, commercial advertisement style, Indian market aesthetic, no text, clean design, product photography"
        
        # Add cultural context based on language
        if language in ['hi', 'mr', 'gu']:  # North Indian languages
            prompt += ", Indian traditional colors, festive atmosphere"
        elif language in ['ta', 'te', 'kn', 'ml']:  # South Indian languages
            prompt += ", South Indian aesthetic, vibrant temple colors"
        
        return prompt
    
    def _generate_base_image(self, prompt: str) -> Image:
        """
        Generate poster image using Pollinations.ai
        """
        encoded_prompt = urllib.parse.quote(prompt)
        image_url = f"{self.api_url}{encoded_prompt}?width=1024&height=1024&nologo=true&enhance=true"
        
        print(f"[Poster] Generating: {prompt[:80]}...")
        
        response = requests.get(image_url, timeout=30)
        
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        else:
            raise Exception(f"Image generation failed: {response.status_code}")
    
    def _add_multilingual_overlays(
        self,
        image: Image,
        product_info: Dict,
        discount_info: Dict,
        language: str
    ) -> Image:
        """
        Add text overlays in detected language
        """
        draw = ImageDraw.Draw(image)
        
        # Get translations
        trans = self.translations.get(language, self.translations['en'])
        
        # Try to load language-specific font
        try:
            font_path = self.font_paths.get(language, 'Arial-Bold.ttf')
            title_font = ImageFont.truetype(font_path, 80)
            price_font = ImageFont.truetype(font_path, 120)
            discount_font = ImageFont.truetype(font_path, 70)
            small_font = ImageFont.truetype(font_path, 50)
        except:
            # Fallback to default
            title_font = ImageFont.load_default()
            price_font = ImageFont.load_default()
            discount_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add product name (top center)
        product_name = product_info.get('product_name', 'Product')
        self._draw_text_with_outline(
            draw, (512, 80), product_name,
            title_font, 'white', 'black', anchor='mm'
        )
        
        # Add discount badge (top-right) if discount > 0
        if discount_info['discount_percentage'] > 0:
            self._draw_discount_badge(
                image, draw,
                discount_info['discount_percentage'],
                trans['off'],
                discount_font
            )
        
        # Add price section (bottom)
        self._draw_price_section(
            draw,
            discount_info,
            price_font,
            small_font,
            trans
        )
        
        # Add business name (bottom-right)
        business_name = product_info.get('business_name', '')
        if business_name:
            self._draw_text_with_outline(
                draw, (900, 980), business_name,
                small_font, 'white', 'black', anchor='rm'
            )
        
        return image
    
    def _draw_text_with_outline(
        self,
        draw,
        position,
        text,
        font,
        fill_color,
        outline_color,
        anchor='lt'
    ):
        """
        Draw text with outline for better visibility
        """
        x, y = position
        
        # Draw outline
        for adj_x in range(-3, 4):
            for adj_y in range(-3, 4):
                draw.text(
                    (x + adj_x, y + adj_y),
                    text,
                    font=font,
                    fill=outline_color,
                    anchor=anchor
                )
        
        # Draw main text
        draw.text(position, text, font=font, fill=fill_color, anchor=anchor)
    
    def _draw_discount_badge(self, image, draw, discount_pct, off_text, font):
        """
        Draw circular discount badge
        """
        # Create circular badge
        badge_center = (850, 150)
        badge_radius = 100
        
        # Draw circle
        draw.ellipse(
            [
                badge_center[0] - badge_radius,
                badge_center[1] - badge_radius,
                badge_center[0] + badge_radius,
                badge_center[1] + badge_radius
            ],
            fill='red',
            outline='white',
            width=5
        )
        
        # Draw discount text
        badge_text = f"{discount_pct}%"
        self._draw_text_with_outline(
            draw,
            (badge_center[0], badge_center[1] - 20),
            badge_text,
            font,
            'white',
            'darkred',
            anchor='mm'
        )
        
        # Draw "OFF" text
        small_font = ImageFont.load_default()
        self._draw_text_with_outline(
            draw,
            (badge_center[0], badge_center[1] + 30),
            off_text,
            small_font,
            'white',
            'darkred',
            anchor='mm'
        )
    
    def _draw_price_section(self, draw, discount_info, price_font, small_font, trans):
        """
        Draw price section with original and discounted prices
        """
        if discount_info['discount_percentage'] > 0:
            # Strike-through original price
            original_text = f"₹{discount_info['original_price']}"
            self._draw_text_with_outline(
                draw, (100, 850), original_text,
                price_font, 'gray', 'black'
            )
            
            # Strike-through line
            draw.line([(100, 900), (400, 900)], fill='red', width=8)
            
            # New price (larger, green)
            new_price_text = f"₹{discount_info['discounted_price']}"
            self._draw_text_with_outline(
                draw, (100, 950), new_price_text,
                price_font, '#00ff00', 'darkgreen'
            )
            
            # Savings text
            savings_text = f"{trans['save']} ₹{discount_info['savings']}"
            self._draw_text_with_outline(
                draw, (100, 1000), savings_text,
                small_font, 'yellow', 'black'
            )
        else:
            # Just show price
            price_text = f"₹{discount_info['original_price']}"
            self._draw_text_with_outline(
                draw, (100, 920), price_text,
                price_font, 'white', 'black'
            )


# Singleton instance
multilingual_poster_gen = MultilingualPosterGenerator()
