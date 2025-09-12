import json
import re
import sqlite3
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from fuzzywuzzy import fuzz
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProductMatch:
    name: str
    carbon_footprint: float
    confidence: float
    category: str
    source: str
    unit: str = "kg"
    alternatives: List[str] = None

class CarbonFootprintAnalyzer:
    def __init__(self, db_path: str = "carbon_database.db"):
        self.db_path = db_path
        self.carbon_database = {}
        self.category_multipliers = {}
        self.sustainable_alternatives = {}
        
        # Initialize databases
        self._initialize_carbon_database()
        self._initialize_alternatives_database()
        self._setup_sqlite_db()
    
    def _initialize_carbon_database(self):
        """Initialize comprehensive carbon footprint database"""
        self.carbon_database = {
            # Fruits (kg CO2 per kg)
            'apple': {'carbon': 0.33, 'category': 'fruit', 'unit': 'kg', 'source': 'FAO'},
            'banana': {'carbon': 0.48, 'category': 'fruit', 'unit': 'kg', 'source': 'FAO'},
            'orange': {'carbon': 0.31, 'category': 'fruit', 'unit': 'kg', 'source': 'FAO'},
            'grapes': {'carbon': 0.58, 'category': 'fruit', 'unit': 'kg', 'source': 'FAO'},
            'strawberry': {'carbon': 0.85, 'category': 'fruit', 'unit': 'kg', 'source': 'FAO'},
            'avocado': {'carbon': 0.42, 'category': 'fruit', 'unit': 'kg', 'source': 'FAO'},
            'lemon': {'carbon': 0.35, 'category': 'fruit', 'unit': 'kg', 'source': 'FAO'},
            'peach': {'carbon': 0.29, 'category': 'fruit', 'unit': 'kg', 'source': 'FAO'},
            
            # Vegetables (kg CO2 per kg)
            'tomato': {'carbon': 0.84, 'category': 'vegetable', 'unit': 'kg', 'source': 'FAO'},
            'potato': {'carbon': 0.29, 'category': 'vegetable', 'unit': 'kg', 'source': 'FAO'},
            'onion': {'carbon': 0.25, 'category': 'vegetable', 'unit': 'kg', 'source': 'FAO'},
            'carrot': {'carbon': 0.23, 'category': 'vegetable', 'unit': 'kg', 'source': 'FAO'},
            'lettuce': {'carbon': 0.62, 'category': 'vegetable', 'unit': 'kg', 'source': 'FAO'},
            'broccoli': {'carbon': 0.41, 'category': 'vegetable', 'unit': 'kg', 'source': 'FAO'},
            'spinach': {'carbon': 0.71, 'category': 'vegetable', 'unit': 'kg', 'source': 'FAO'},
            'bell pepper': {'carbon': 0.73, 'category': 'vegetable', 'unit': 'kg', 'source': 'FAO'},
            'cucumber': {'carbon': 0.52, 'category': 'vegetable', 'unit': 'kg', 'source': 'FAO'},
            
            # Meat & Poultry (kg CO2 per kg)
            'beef': {'carbon': 13.6, 'category': 'meat', 'unit': 'kg', 'source': 'EPA'},
            'lamb': {'carbon': 12.1, 'category': 'meat', 'unit': 'kg', 'source': 'EPA'},
            'pork': {'carbon': 4.62, 'category': 'meat', 'unit': 'kg', 'source': 'EPA'},
            'chicken': {'carbon': 2.33, 'category': 'meat', 'unit': 'kg', 'source': 'EPA'},
            'turkey': {'carbon': 2.08, 'category': 'meat', 'unit': 'kg', 'source': 'EPA'},
            'duck': {'carbon': 3.12, 'category': 'meat', 'unit': 'kg', 'source': 'EPA'},
            'ground beef': {'carbon': 13.6, 'category': 'meat', 'unit': 'kg', 'source': 'EPA'},
            'chicken breast': {'carbon': 2.33, 'category': 'meat', 'unit': 'kg', 'source': 'EPA'},
            
            # Seafood (kg CO2 per kg)
            'salmon': {'carbon': 2.9, 'category': 'seafood', 'unit': 'kg', 'source': 'Seafood Watch'},
            'tuna': {'carbon': 3.8, 'category': 'seafood', 'unit': 'kg', 'source': 'Seafood Watch'},
            'cod': {'carbon': 2.1, 'category': 'seafood', 'unit': 'kg', 'source': 'Seafood Watch'},
            'shrimp': {'carbon': 7.2, 'category': 'seafood', 'unit': 'kg', 'source': 'Seafood Watch'},
            'crab': {'carbon': 8.9, 'category': 'seafood', 'unit': 'kg', 'source': 'Seafood Watch'},
            'lobster': {'carbon': 11.3, 'category': 'seafood', 'unit': 'kg', 'source': 'Seafood Watch'},
            
            # Dairy (kg CO2 per kg or liter)
            'milk': {'carbon': 1.19, 'category': 'dairy', 'unit': 'liter', 'source': 'FAO'},
            'cheese': {'carbon': 5.15, 'category': 'dairy', 'unit': 'kg', 'source': 'FAO'},
            'yogurt': {'carbon': 0.91, 'category': 'dairy', 'unit': 'kg', 'source': 'FAO'},
            'butter': {'carbon': 9.02, 'category': 'dairy', 'unit': 'kg', 'source': 'FAO'},
            'cream': {'carbon': 4.23, 'category': 'dairy', 'unit': 'liter', 'source': 'FAO'},
            'ice cream': {'carbon': 3.8, 'category': 'dairy', 'unit': 'kg', 'source': 'FAO'},
            'cheddar': {'carbon': 5.15, 'category': 'dairy', 'unit': 'kg', 'source': 'FAO'},
            'mozzarella': {'carbon': 4.89, 'category': 'dairy', 'unit': 'kg', 'source': 'FAO'},
            
            # Grains & Cereals (kg CO2 per kg)
            'rice': {'carbon': 1.21, 'category': 'grain', 'unit': 'kg', 'source': 'FAO'},
            'wheat': {'carbon': 0.67, 'category': 'grain', 'unit': 'kg', 'source': 'FAO'},
            'bread': {'carbon': 0.77, 'category': 'grain', 'unit': 'kg', 'source': 'FAO'},
            'pasta': {'carbon': 0.87, 'category': 'grain', 'unit': 'kg', 'source': 'FAO'},
            'oats': {'carbon': 0.94, 'category': 'grain', 'unit': 'kg', 'source': 'FAO'},
            'barley': {'carbon': 0.82, 'category': 'grain', 'unit': 'kg', 'source': 'FAO'},
            'quinoa': {'carbon': 1.85, 'category': 'grain', 'unit': 'kg', 'source': 'FAO'},
            'cereal': {'carbon': 0.51, 'category': 'grain', 'unit': 'kg', 'source': 'FAO'},
            
            # Legumes & Nuts (kg CO2 per kg)
            'beans': {'carbon': 0.43, 'category': 'legume', 'unit': 'kg', 'source': 'FAO'},
            'lentils': {'carbon': 0.43, 'category': 'legume', 'unit': 'kg', 'source': 'FAO'},
            'chickpeas': {'carbon': 0.58, 'category': 'legume', 'unit': 'kg', 'source': 'FAO'},
            'almonds': {'carbon': 8.8, 'category': 'nuts', 'unit': 'kg', 'source': 'Water Footprint Network'},
            'walnuts': {'carbon': 2.61, 'category': 'nuts', 'unit': 'kg', 'source': 'Water Footprint Network'},
            'peanuts': {'carbon': 0.42, 'category': 'nuts', 'unit': 'kg', 'source': 'Water Footprint Network'},
            
            # Eggs & Protein (kg CO2 per unit or kg)
            'eggs': {'carbon': 1.91, 'category': 'protein', 'unit': 'dozen', 'source': 'EPA'},
            'tofu': {'carbon': 0.88, 'category': 'protein', 'unit': 'kg', 'source': 'FAO'},
            'tempeh': {'carbon': 0.71, 'category': 'protein', 'unit': 'kg', 'source': 'FAO'},
            
            # Beverages (kg CO2 per liter)
            'coffee': {'carbon': 0.28, 'category': 'beverage', 'unit': 'cup', 'source': 'Carbon Trust'},
            'tea': {'carbon': 0.02, 'category': 'beverage', 'unit': 'cup', 'source': 'Carbon Trust'},
            'wine': {'carbon': 1.79, 'category': 'beverage', 'unit': 'bottle', 'source': 'Wine Institute'},
            'beer': {'carbon': 0.74, 'category': 'beverage', 'unit': 'bottle', 'source': 'Brewers Association'},
            'orange juice': {'carbon': 0.34, 'category': 'beverage', 'unit': 'liter', 'source': 'FAO'},
            'soda': {'carbon': 0.33, 'category': 'beverage', 'unit': 'liter', 'source': 'EPA'},
            
            # Processed Foods (kg CO2 per kg)
            'chocolate': {'carbon': 2.3, 'category': 'processed', 'unit': 'kg', 'source': 'Cocoa Industry'},
            'chips': {'carbon': 1.87, 'category': 'processed', 'unit': 'kg', 'source': 'Snack Food Association'},
            'cookies': {'carbon': 1.64, 'category': 'processed', 'unit': 'kg', 'source': 'Food Industry'},
            'frozen pizza': {'carbon': 1.35, 'category': 'processed', 'unit': 'kg', 'source': 'Food Industry'},
            'canned tomatoes': {'carbon': 0.67, 'category': 'processed', 'unit': 'kg', 'source': 'Canning Industry'}
        }
        
        # Category-based multipliers for organic/local variations
        self.category_multipliers = {
            'organic': 0.85,  # Generally 15% less carbon footprint
            'local': 0.75,   # 25% less due to reduced transport
            'grass-fed': 1.1, # Slightly higher for grass-fed meat
            'free-range': 0.95, # Slightly better for free-range
            'wild-caught': 0.8, # Better than farmed seafood
            'processed': 1.25,  # 25% higher for processed foods
            'imported': 1.4    # 40% higher for imported goods
        }
    
    def _initialize_alternatives_database(self):
        """Initialize sustainable alternatives database"""
        self.sustainable_alternatives = {
            'beef': [
                {'name': 'Beyond Meat Plant-Based Patties', 'carbon_reduction': 0.89, 'description': 'Plant-based protein with 89% less carbon footprint'},
                {'name': 'Organic Chicken Breast', 'carbon_reduction': 0.83, 'description': 'Lower carbon footprint poultry option'},
                {'name': 'Black Bean Burgers', 'carbon_reduction': 0.97, 'description': 'Legume-based protein alternative'},
                {'name': 'Portobello Mushrooms', 'carbon_reduction': 0.95, 'description': 'Meaty texture with minimal carbon impact'}
            ],
            'lamb': [
                {'name': 'Organic Turkey', 'carbon_reduction': 0.83, 'description': 'Lower carbon footprint poultry'},
                {'name': 'Plant-Based Protein', 'carbon_reduction': 0.93, 'description': 'Complete protein with minimal emissions'}
            ],
            'cheese': [
                {'name': 'Cashew-Based Cheese', 'carbon_reduction': 0.78, 'description': 'Nut-based dairy alternative'},
                {'name': 'Nutritional Yeast', 'carbon_reduction': 0.92, 'description': 'Cheesy flavor with B-vitamins'},
                {'name': 'Almond Cheese', 'carbon_reduction': 0.71, 'description': 'Tree nut-based alternative'}
            ],
            'milk': [
                {'name': 'Oat Milk', 'carbon_reduction': 0.71, 'description': 'Grain-based milk with creamy texture'},
                {'name': 'Almond Milk', 'carbon_reduction': 0.65, 'description': 'Light, nutty flavor profile'},
                {'name': 'Soy Milk', 'carbon_reduction': 0.63, 'description': 'High protein plant-based milk'}
            ],
            'chicken': [
                {'name': 'Organic Tofu', 'carbon_reduction': 0.62, 'description': 'Versatile soy-based protein'},
                {'name': 'Tempeh', 'carbon_reduction': 0.70, 'description': 'Fermented soy with nutty flavor'},
                {'name': 'Seitan', 'carbon_reduction': 0.75, 'description': 'Wheat-based protein with meaty texture'}
            ],
            'salmon': [
                {'name': 'Plant-Based Fish Fillets', 'carbon_reduction': 0.85, 'description': 'Algae-based seafood alternative'},
                {'name': 'Hemp Seed Protein', 'carbon_reduction': 0.91, 'description': 'Complete protein with omega-3s'},
                {'name': 'Flaxseed Oil Supplements', 'carbon_reduction': 0.95, 'description': 'Omega-3 fatty acids from plants'}
            ],
            'rice': [
                {'name': 'Quinoa', 'carbon_reduction': 0.15, 'description': 'Higher protein grain alternative'},
                {'name': 'Barley', 'carbon_reduction': 0.32, 'description': 'Lower carbon grain with fiber'},
                {'name': 'Cauliflower Rice', 'carbon_reduction': 0.76, 'description': 'Vegetable-based rice substitute'}
            ],
            'pasta': [
                {'name': 'Lentil Pasta', 'carbon_reduction': 0.51, 'description': 'High protein legume-based noodles'},
                {'name': 'Chickpea Pasta', 'carbon_reduction': 0.33, 'description': 'Gluten-free high-protein alternative'},
                {'name': 'Zucchini Noodles', 'carbon_reduction': 0.83, 'description': 'Fresh vegetable pasta substitute'}
            ]
        }
    
    def _setup_sqlite_db(self):
        """Initialize SQLite database for caching and analytics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS carbon_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    store_name TEXT,
                    total_carbon REAL,
                    total_cost REAL,
                    product_count INTEGER,
                    analysis_data TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS product_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_name TEXT,
                    matched_name TEXT,
                    carbon_footprint REAL,
                    confidence REAL,
                    category TEXT,
                    timestamp TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("SQLite database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def fuzzy_match_product(self, product_name: str, threshold: float = 70) -> Optional[ProductMatch]:
        """Find the best match for a product using fuzzy matching"""
        best_match = None
        best_score = 0
        
        # Clean the product name
        cleaned_name = self._clean_product_name(product_name)
        
        # Search through carbon database
        for db_name, data in self.carbon_database.items():
            # Calculate fuzzy match score
            score = fuzz.partial_ratio(cleaned_name.lower(), db_name.lower())
            
            # Also check token sort ratio for better matching
            token_score = fuzz.token_sort_ratio(cleaned_name.lower(), db_name.lower())
            final_score = max(score, token_score)
            
            if final_score > best_score and final_score >= threshold:
                best_score = final_score
                best_match = ProductMatch(
                    name=db_name,
                    carbon_footprint=data['carbon'],
                    confidence=final_score / 100.0,
                    category=data['category'],
                    source=data['source'],
                    unit=data['unit'],
                    alternatives=self.sustainable_alternatives.get(db_name, [])
                )
        
        return best_match
    
    def _clean_product_name(self, name: str) -> str:
        """Clean and normalize product names for better matching"""
        # Remove common prefixes and suffixes
        prefixes_to_remove = ['organic', 'fresh', 'frozen', 'canned', 'whole', 'raw', 'cooked']
        suffixes_to_remove = ['lb', 'lbs', 'oz', 'kg', 'g', 'count', 'ct', 'pack']
        
        cleaned = name.lower().strip()
        
        # Remove quantity patterns
        cleaned = re.sub(r'\d+(?:\.\d+)?\s*(?:lbs?|oz|kg|g|count|ct)', '', cleaned)
        cleaned = re.sub(r'\(\d+(?:\.\d+)?\s*(?:lbs?|oz|kg|g|count|ct)\)', '', cleaned)
        
        # Remove brand names (common patterns)
        cleaned = re.sub(r'^(organic|fresh|frozen|whole|raw)\s+', '', cleaned)
        
        # Remove special characters and extra spaces
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def calculate_carbon_footprint(self, products: List[Dict], store_info: Dict = None) -> Dict:
        """Calculate total carbon footprint for a list of products"""
        results = {
            'total_carbon': 0.0,
            'total_cost': 0.0,
            'matched_products': [],
            'unmatched_products': [],
            'statistics': {},
            'recommendations': [],
            'environmental_score': 'C'
        }
        
        try:
            for product in products:
                product_result = self._analyze_single_product(product, store_info)
                
                if product_result['matched']:
                    results['matched_products'].append(product_result)
                    results['total_carbon'] += product_result['carbon_footprint']
                else:
                    results['unmatched_products'].append(product_result)
                
                results['total_cost'] += product.get('price', 0)
            
            # Calculate statistics
            results['statistics'] = self._calculate_statistics(results)
            
            # Generate recommendations (pass results to avoid undefined variable error)
            results['recommendations'] = self._generate_recommendations(results['matched_products'], results)
            
            # Calculate environmental score
            results['environmental_score'] = self._calculate_environmental_score(
                results['total_carbon'], len(results['matched_products'])
            )
            
            # Cache results
            self._cache_analysis(results, store_info)
            
            logger.info(f"Carbon analysis complete: {results['total_carbon']:.2f} kg CO2")
            return results
            
        except Exception as e:
            logger.error(f"Carbon calculation failed: {e}")
            return results
    
    def _analyze_single_product(self, product: Dict, store_info: Dict = None) -> Dict:
        """Analyze a single product for carbon footprint"""
        result = {
            'original_name': product.get('name', ''),
            'price': product.get('price', 0),
            'quantity': product.get('quantity', 1),
            'matched': False,
            'carbon_footprint': 0.0,
            'carbon_per_dollar': 0.0,
            'match_confidence': 0.0,
            'category': 'unknown',
            'alternatives': [],
            'modifiers': []
        }
        
        # Find best match
        match = self.fuzzy_match_product(product.get('name', ''))
        
        if match:
            result['matched'] = True
            result['matched_name'] = match.name
            result['match_confidence'] = match.confidence
            result['category'] = match.category
            result['alternatives'] = match.alternatives or []
            
            # Calculate carbon footprint
            base_carbon = match.carbon_footprint * result['quantity']
            
            # Apply modifiers based on product name
            modified_carbon = self._apply_modifiers(base_carbon, product.get('name', ''))
            result['carbon_footprint'] = modified_carbon
            result['modifiers'] = self._get_applicable_modifiers(product.get('name', ''))
            
            # Calculate carbon per dollar
            if result['price'] > 0:
                result['carbon_per_dollar'] = result['carbon_footprint'] / result['price']
            
            # Cache the match
            self._cache_product_match(result)
        
        return result
    
    def _apply_modifiers(self, base_carbon: float, product_name: str) -> float:
        """Apply carbon footprint modifiers based on product characteristics"""
        modified_carbon = base_carbon
        
        product_lower = product_name.lower()
        
        # Apply modifiers
        for modifier, multiplier in self.category_multipliers.items():
            if modifier in product_lower:
                modified_carbon *= multiplier
                logger.debug(f"Applied {modifier} modifier: {multiplier}")
        
        return modified_carbon
    
    def _get_applicable_modifiers(self, product_name: str) -> List[str]:
        """Get list of applicable modifiers for a product"""
        modifiers = []
        product_lower = product_name.lower()
        
        for modifier in self.category_multipliers.keys():
            if modifier in product_lower:
                modifiers.append(modifier)
        
        return modifiers
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """Calculate analysis statistics"""
        matched_products = results['matched_products']
        
        if not matched_products:
            return {}
        
        carbon_values = [p['carbon_footprint'] for p in matched_products]
        carbon_per_dollar = [p['carbon_per_dollar'] for p in matched_products if p['carbon_per_dollar'] > 0]
        
        # Category breakdown
        category_carbon = {}
        for product in matched_products:
            category = product['category']
            if category not in category_carbon:
                category_carbon[category] = 0
            category_carbon[category] += product['carbon_footprint']
        
        return {
            'avg_carbon_per_product': np.mean(carbon_values) if carbon_values else 0,
            'highest_carbon_product': max(matched_products, key=lambda x: x['carbon_footprint'])['original_name'] if matched_products else '',
            'lowest_carbon_product': min(matched_products, key=lambda x: x['carbon_footprint'])['original_name'] if matched_products else '',
            'avg_carbon_per_dollar': np.mean(carbon_per_dollar) if carbon_per_dollar else 0,
            'category_breakdown': category_carbon,
            'match_rate': len(matched_products) / (len(matched_products) + len(results['unmatched_products'])) * 100
        }
    
    def _generate_recommendations(self, matched_products: List[Dict], results: Dict = None) -> List[Dict]:
        """Generate sustainability recommendations"""
        recommendations = []
        
        # Find high-carbon products and suggest alternatives
        high_carbon_products = sorted(matched_products, key=lambda x: x['carbon_footprint'], reverse=True)[:3]
        
        for product in high_carbon_products:
            if product['alternatives']:
                for alt in product['alternatives'][:2]:  # Top 2 alternatives
                    potential_savings = product['carbon_footprint'] * alt['carbon_reduction']
                    recommendations.append({
                        'type': 'product_swap',
                        'original_product': product['original_name'],
                        'alternative': alt['name'],
                        'description': alt['description'],
                        'carbon_savings': potential_savings,
                        'percentage_reduction': alt['carbon_reduction'] * 100
                    })
        
        # Add general recommendations
        if recommendations and results:
            total_carbon = results.get('total_carbon', 0)
            recommendations.append({
                'type': 'general',
                'title': 'Shop Local',
                'description': 'Choose locally sourced products to reduce transportation emissions',
                'potential_savings': total_carbon * 0.25
            })
            
            recommendations.append({
                'type': 'general',
                'title': 'Organic Options',
                'description': 'Consider organic alternatives which typically have 15% lower carbon footprint',
                'potential_savings': total_carbon * 0.15
            })
        
        return recommendations
    
    def _calculate_environmental_score(self, total_carbon: float, product_count: int) -> str:
        """Calculate environmental score (A+ to F)"""
        if product_count == 0:
            return 'N/A'
        
        avg_carbon = total_carbon / product_count
        
        if avg_carbon < 1.0:
            return 'A+'
        elif avg_carbon < 1.5:
            return 'A'
        elif avg_carbon < 2.5:
            return 'B'
        elif avg_carbon < 4.0:
            return 'C'
        elif avg_carbon < 6.0:
            return 'D'
        else:
            return 'F'
    
    def _cache_analysis(self, results: Dict, store_info: Dict = None):
        """Cache analysis results in SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO carbon_analyses 
                (timestamp, store_name, total_carbon, total_cost, product_count, analysis_data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                store_info.get('store_name', '') if store_info else '',
                results['total_carbon'],
                results['total_cost'],
                len(results['matched_products']),
                json.dumps(results, default=str)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Caching failed: {e}")
    
    def _cache_product_match(self, result: Dict):
        """Cache individual product matches"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO product_matches 
                (original_name, matched_name, carbon_footprint, confidence, category, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                result['original_name'],
                result.get('matched_name', ''),
                result['carbon_footprint'],
                result['match_confidence'],
                result['category'],
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Product match caching failed: {e}")
    
    def get_analytics(self, days: int = 30) -> Dict:
        """Get analytics from cached data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent analyses
            cursor.execute('''
                SELECT * FROM carbon_analyses 
                WHERE datetime(timestamp) >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days))
            
            analyses = cursor.fetchall()
            
            # Calculate trends
            if analyses:
                total_carbon_saved = sum(row[2] for row in analyses)  # total_carbon column
                avg_carbon_per_receipt = total_carbon_saved / len(analyses)
                
                analytics = {
                    'total_analyses': len(analyses),
                    'total_carbon_analyzed': total_carbon_saved,
                    'avg_carbon_per_receipt': avg_carbon_per_receipt,
                    'trend_data': analyses[:10]  # Last 10 analyses
                }
            else:
                analytics = {'message': 'No data available for the specified period'}
            
            conn.close()
            return analytics
            
        except Exception as e:
            logger.error(f"Analytics query failed: {e}")
            return {'error': str(e)}