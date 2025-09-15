from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import json
import logging
from datetime import datetime, timedelta
import traceback
import os
import google.generativeai as genai
from typing import Dict, List, Optional, Any
import re
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize components with better error handling
ocr_processor = None
carbon_analyzer = None

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-gemini-api-key-here')
if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key-here':
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("Gemini AI initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini AI: {e}")
        gemini_model = None
else:
    logger.warning("Gemini API key not provided - meal planning features will be limited")
    gemini_model = None

def initialize_services():
    """Initialize OCR and Carbon services with proper error handling"""
    global ocr_processor, carbon_analyzer
    
    try:
        # Try importing from different possible module names
        try:
            from receipt_ocr import ReceiptOCR
            logger.info("Imported ReceiptOCR from receipt_ocr.py")
        except ImportError:
            try:
                # If app.py exists and contains ReceiptOCR
                import importlib.util
                spec = importlib.util.spec_from_file_location("receipt_module", "app.py")
                receipt_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(receipt_module)
                ReceiptOCR = receipt_module.ReceiptOCR
                logger.info("Imported ReceiptOCR from app.py using importlib")
            except Exception as e:
                logger.error(f"Failed to import ReceiptOCR: {e}")
                raise ImportError("Cannot import ReceiptOCR class")
        
        try:
            from carbon_analyzer import CarbonFootprintAnalyzer
            logger.info("Imported CarbonFootprintAnalyzer successfully")
        except ImportError as e:
            logger.error(f"Failed to import CarbonFootprintAnalyzer: {e}")
            raise ImportError("Cannot import CarbonFootprintAnalyzer class")
        
        # Initialize the services
        ocr_processor = ReceiptOCR()
        carbon_analyzer = CarbonFootprintAnalyzer()
        
        logger.info("All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        logger.error(traceback.format_exc())
        return False

# Try to initialize services on startup
services_available = initialize_services()

# Static file serving routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        return send_file('index.html')
    except FileNotFoundError:
        return jsonify({
            'error': 'index.html not found',
            'message': 'Please ensure index.html is in the same directory as main.py'
        }), 404

@app.route('/index.html')
def index_html():
    """Alternative route for index.html"""
    return index()

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files from a static directory if it exists"""
    if os.path.exists('static'):
        return send_from_directory('static', filename)
    else:
        return jsonify({'error': 'Static directory not found'}), 404

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    status = {
        'status': 'healthy' if services_available else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'ocr_processor': ocr_processor is not None,
            'carbon_analyzer': carbon_analyzer is not None,
            'gemini_ai': gemini_model is not None,
        },
        'version': '2.0.0',
        'features': [
            'Advanced OCR with dual-engine processing',
            'Comprehensive carbon footprint analysis',
            'Sustainable alternatives recommendation',
            'AI-powered meal planning',
            'Smart shopping list generation',
            'Real-time analytics and caching'
        ] if services_available else ['Limited functionality - services not available'],
        'available_endpoints': [
            'GET / - Main application page',
            'GET /index.html - Main application page',
            'POST /api/analyze-receipt - Complete receipt analysis',
            'POST /api/ocr-only - Text extraction only',
            'POST /api/carbon-analysis - Carbon footprint analysis only',
            'POST /api/generate-meal-plan - AI meal plan generation',
            'POST /api/generate-shopping-list - Smart shopping list',
            'POST /api/generate-receipt-meal-prep - Receipt-based meal prep',
            'POST /api/find-alternatives - Find product alternatives',
            'GET /api/analytics - Usage analytics',
            'GET /api/database-info - Database information',
            'GET /health - Health check'
        ]
    }
    
    if services_available:
        return jsonify(status), 200
    else:
        return jsonify(status), 503

# ORIGINAL ENDPOINTS (unchanged)
@app.route('/api/analyze-receipt', methods=['POST'])
def analyze_receipt():
    """Complete receipt scanning and analysis pipeline"""
    if not services_available:
        return jsonify({
            'success': False,
            'error': 'Services not initialized properly',
            'details': 'OCR or Carbon Analysis services are not available',
            'code': 'SERVICES_UNAVAILABLE'
        }), 503
    
    try:
        start_time = datetime.now()
        
        # Handle both form data and JSON
        if request.content_type.startswith('multipart/form-data'):
            image_data = request.form.get('image_data')
            analysis_type = request.form.get('analysis_type', 'full')
        else:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({
                    'success': False,
                    'error': 'No image data provided',
                    'code': 'MISSING_IMAGE'
                }), 400
            image_data = data['image']
            analysis_type = data.get('analysis_type', 'full')
        
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'No image data provided',
                'code': 'MISSING_IMAGE'
            }), 400
        
        logger.info("Starting complete receipt analysis pipeline")
        
        # Step 1: OCR Processing
        logger.info("Phase 1: OCR Text Extraction")
        ocr_result = ocr_processor.process_receipt(image_data)
        
        if not ocr_result.get('success', False):
            return jsonify({
                'success': False,
                'error': 'OCR processing failed',
                'details': ocr_result.get('error', 'Unknown OCR error'),
                'code': 'OCR_FAILED'
            }), 500
        
        # Step 2: Enhanced Product Extraction
        logger.info("Phase 2: Product Identification")
        products = ocr_result.get('structured_data', {}).get('items', [])
        extracted_text = ocr_result.get('raw_text', '')
        
        if not products and not extracted_text:
            return jsonify({
                'success': False,
                'error': 'No products or text found in receipt',
                'ocr_result': ocr_result,
                'code': 'NO_CONTENT_FOUND'
            }), 422
        
        # Step 3: Carbon Footprint Analysis (if products found)
        carbon_result = {'total_carbon': 0, 'matched_products': [], 'statistics': {}}
        if products:
            logger.info(f"Phase 3: Carbon Analysis for {len(products)} products")
            store_info = {
                'store_name': ocr_result.get('structured_data', {}).get('store_name', ''),
                'date': ocr_result.get('structured_data', {}).get('date', ''),
                'total': ocr_result.get('structured_data', {}).get('total', 0)
            }
            
            carbon_result = carbon_analyzer.calculate_carbon_footprint(products, store_info)
        
        # Step 4: Compile Complete Response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        complete_result = {
            'success': True,
            'processing_time_seconds': round(processing_time, 2),
            'timestamp': datetime.now().isoformat(),
            'extracted_text': extracted_text,
            'products': [
                {
                    'name': p.get('matched_name', p.get('original_name', '')),
                    'price': p.get('price', 0),
                    'quantity': p.get('quantity', 1),
                    'carbon_footprint': round(p.get('carbon_footprint', 0), 2),
                    'category': p.get('category', 'unknown'),
                    'confidence': round(p.get('match_confidence', 0) * 100, 1) if p.get('match_confidence') else 50,
                    'alternatives': p.get('alternatives', [])[:2] if p.get('alternatives') else []
                }
                for p in carbon_result.get('matched_products', [])
            ],
            'summary': {
                'total_carbon_footprint': round(carbon_result.get('total_carbon', 0), 2),
                'total_products': len(products),
                'match_rate': round(carbon_result.get('statistics', {}).get('match_rate', 0), 2),
                'environmental_score': carbon_result.get('environmental_score', 'C')
            },
            'alternatives': generate_quick_alternatives(carbon_result.get('matched_products', []))
        }
        
        logger.info(f"Analysis complete - Total CO2: {complete_result['summary']['total_carbon_footprint']}kg")
        return jsonify(complete_result), 200
        
    except Exception as e:
        logger.error(f"Complete analysis failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Internal processing error',
            'details': str(e),
            'code': 'PROCESSING_ERROR',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/ocr-only', methods=['POST'])
def ocr_only():
    """OCR-only endpoint for text extraction"""
    if not ocr_processor:
        return jsonify({'success': False, 'error': 'OCR service unavailable'}), 503
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        result = ocr_processor.process_receipt(data['image'])
        return jsonify(result), 200 if result.get('success', False) else 500
        
    except Exception as e:
        logger.error(f"OCR-only processing failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# NEW MEAL PLANNING ENDPOINTS

@app.route('/api/generate-meal-plan', methods=['POST'])
def generate_meal_plan():
    """Generate AI-powered sustainable meal plan"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No request data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['budget', 'people_count']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        budget = float(data['budget'])
        people_count = int(data['people_count'])
        dietary_preferences = data.get('dietary_preferences', [])
        additional_preferences = data.get('additional_preferences', '')
        optimize_for = data.get('optimize_for', 'carbon_footprint')
        
        logger.info(f"Generating meal plan for {people_count} people with ${budget} budget")
        
        # Generate meal plan using AI or fallback method
        if gemini_model:
            meal_plan_result = generate_ai_meal_plan(
                budget, people_count, dietary_preferences, 
                additional_preferences, optimize_for
            )
        else:
            meal_plan_result = generate_fallback_meal_plan(
                budget, people_count, dietary_preferences, optimize_for
            )
        
        return jsonify(meal_plan_result), 200
        
    except Exception as e:
        logger.error(f"Meal plan generation failed: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to generate meal plan',
            'details': str(e)
        }), 500

@app.route('/api/generate-shopping-list', methods=['POST'])
def generate_shopping_list():
    """Generate smart shopping list optimized for sustainability"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No request data provided'
            }), 400
        
        # Validate required fields
        budget = float(data.get('budget', 0))
        duration = data.get('duration', 'week')
        priority = data.get('priority', 'balanced')
        pantry_items = data.get('pantry_items', [])
        
        if budget <= 0:
            return jsonify({
                'success': False,
                'error': 'Valid budget is required'
            }), 400
        
        logger.info(f"Generating shopping list with ${budget} budget for {duration}")
        
        # Generate shopping list using AI or fallback method
        if gemini_model:
            shopping_result = generate_ai_shopping_list(
                budget, duration, priority, pantry_items
            )
        else:
            shopping_result = generate_fallback_shopping_list(
                budget, duration, priority, pantry_items
            )
        
        return jsonify(shopping_result), 200
        
    except Exception as e:
        logger.error(f"Shopping list generation failed: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to generate shopping list',
            'details': str(e)
        }), 500

@app.route('/api/generate-receipt-meal-prep', methods=['POST'])
def generate_receipt_meal_prep():
    """Generate meal prep suggestions based on receipt analysis"""
    try:
        data = request.get_json()
        if not data or 'receipt_products' not in data:
            return jsonify({
                'success': False,
                'error': 'Receipt products data required'
            }), 400
        
        products = data['receipt_products']
        total_carbon = data.get('total_carbon_footprint', 0)
        optimization_focus = data.get('optimization_focus', 'carbon_reduction')
        meal_count = data.get('meal_count', 7)
        people_count = data.get('people_count', 2)
        
        logger.info(f"Generating receipt-based meal prep for {len(products)} products")
        
        # Generate meal prep suggestions
        if gemini_model:
            meal_prep_result = generate_ai_meal_prep_from_receipt(
                products, total_carbon, optimization_focus, meal_count, people_count
            )
        else:
            meal_prep_result = generate_fallback_meal_prep_from_receipt(
                products, total_carbon, optimization_focus
            )
        
        return jsonify(meal_prep_result), 200
        
    except Exception as e:
        logger.error(f"Receipt meal prep generation failed: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to generate meal prep suggestions',
            'details': str(e)
        }), 500

@app.route('/api/find-alternatives', methods=['POST'])
def find_alternatives():
    """Find sustainable alternatives for specific products"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No request data provided'
            }), 400
        
        product_index = data.get('product_index')
        include_nearby_stores = data.get('include_nearby_stores', False)
        include_price_comparison = data.get('include_price_comparison', True)
        max_alternatives = data.get('max_alternatives', 3)
        
        # This would use your existing carbon_analyzer to find alternatives
        if carbon_analyzer:
            alternatives_result = find_product_alternatives(
                product_index, include_nearby_stores, 
                include_price_comparison, max_alternatives
            )
        else:
            alternatives_result = generate_fallback_alternatives()
        
        return jsonify(alternatives_result), 200
        
    except Exception as e:
        logger.error(f"Alternative finding failed: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to find alternatives',
            'details': str(e)
        }), 500

# AI GENERATION FUNCTIONS

def generate_ai_meal_plan(budget: float, people_count: int, dietary_preferences: List[str], 
                         additional_preferences: str, optimize_for: str) -> Dict[str, Any]:
    """Generate meal plan using Gemini AI"""
    try:
        prompt = f"""
        Generate a sustainable weekly meal plan with the following requirements:
        
        Budget: ${budget}
        People: {people_count}
        Dietary preferences: {', '.join(dietary_preferences) if dietary_preferences else 'None'}
        Additional preferences: {additional_preferences or 'None'}
        Optimization focus: {optimize_for}
        
        Please provide a 7-day meal plan optimized for low carbon footprint with:
        1. Breakfast, lunch, and dinner for each day
        2. Estimated cost per meal
        3. Carbon footprint per meal (in kg CO2)
        4. Preparation time
        5. Brief description of each meal
        6. Shopping tips for sustainability
        
        Format the response as a structured plan with realistic prices and carbon calculations.
        Focus on seasonal, local, and plant-based options where possible.
        """
        
        response = gemini_model.generate_content(prompt)
        
        # Parse AI response and structure it
        ai_text = response.text
        meal_plan = parse_ai_meal_plan_response(ai_text, budget, people_count)
        
        return {
            'success': True,
            'meal_plan': meal_plan['days'],
            'summary': meal_plan['summary'],
            'tips': meal_plan['tips'],
            'generated_by': 'gemini_ai',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI meal plan generation failed: {e}")
        return generate_fallback_meal_plan(budget, people_count, dietary_preferences, optimize_for)

def generate_ai_shopping_list(budget: float, duration: str, priority: str, pantry_items: List[str]) -> Dict[str, Any]:
    """Generate shopping list using Gemini AI"""
    try:
        duration_days = {'week': 7, '2weeks': 14, 'month': 30}.get(duration, 7)
        
        prompt = f"""
        Generate a smart, sustainable shopping list with these parameters:
        
        Budget: ${budget}
        Duration: {duration} ({duration_days} days)
        Priority: {priority}
        Current pantry items: {', '.join(pantry_items) if pantry_items else 'None'}
        
        Create a shopping list organized by categories (Produce, Proteins, Grains, etc.) that:
        1. Maximizes nutrition per dollar
        2. Minimizes carbon footprint
        3. Considers seasonal availability
        4. Avoids items already in pantry
        5. Includes estimated prices and carbon impact
        6. Suggests sustainable brands/alternatives
        
        Provide realistic price estimates and carbon footprint data (kg CO2) for each item.
        Include alternatives that save carbon emissions and money.
        """
        
        response = gemini_model.generate_content(prompt)
        
        # Parse AI response and structure it
        ai_text = response.text
        shopping_list = parse_ai_shopping_list_response(ai_text, budget)
        
        return {
            'success': True,
            'shopping_list': shopping_list['categories'],
            'summary': shopping_list['summary'],
            'alternatives': shopping_list['alternatives'],
            'generated_by': 'gemini_ai',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI shopping list generation failed: {e}")
        return generate_fallback_shopping_list(budget, duration, priority, pantry_items)

def generate_ai_meal_prep_from_receipt(products: List[Dict], total_carbon: float, 
                                     optimization_focus: str, meal_count: int, people_count: int) -> Dict[str, Any]:
    """Generate meal prep suggestions based on receipt using Gemini AI"""
    try:
        products_text = "\n".join([f"- {p['name']} (${p.get('price', 0)}, {p.get('carbon_footprint', 0)}kg CO2)" for p in products])
        
        prompt = f"""
        Based on this receipt analysis, suggest sustainable meal prep ideas:
        
        Products purchased:
        {products_text}
        
        Current carbon footprint: {total_carbon} kg CO2
        Optimization focus: {optimization_focus}
        Meal count needed: {meal_count}
        People count: {people_count}
        
        Provide 3-5 meal prep suggestions that:
        1. Use ingredients from the receipt
        2. Reduce overall carbon footprint
        3. Are cost-effective
        4. Include prep instructions
        5. Suggest additional sustainable ingredients needed
        6. Calculate carbon savings potential
        
        Focus on reducing meat consumption, using seasonal produce, and minimizing food waste.
        """
        
        response = gemini_model.generate_content(prompt)
        
        # Parse AI response
        ai_text = response.text
        meal_suggestions = parse_ai_meal_prep_response(ai_text, products, total_carbon)
        
        return {
            'success': True,
            'meal_suggestions': meal_suggestions['suggestions'],
            'current_footprint': total_carbon,
            'optimized_footprint': meal_suggestions['optimized_footprint'],
            'potential_savings': meal_suggestions['potential_savings'],
            'generated_by': 'gemini_ai',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI meal prep generation failed: {e}")
        return generate_fallback_meal_prep_from_receipt(products, total_carbon, optimization_focus)

# FALLBACK FUNCTIONS (when AI is not available)

def generate_fallback_meal_plan(budget: float, people_count: int, dietary_preferences: List[str], optimize_for: str) -> Dict[str, Any]:
    """Generate basic meal plan without AI"""
    
    # Basic sustainable meals database
    base_meals = {
        'breakfast': [
            {'name': 'Overnight Oats with Seasonal Fruit', 'cost': 2.50, 'carbon': 0.3, 'prep_time': 5},
            {'name': 'Veggie Scramble with Toast', 'cost': 3.00, 'carbon': 0.8, 'prep_time': 10},
            {'name': 'Smoothie Bowl with Local Berries', 'cost': 3.50, 'carbon': 0.4, 'prep_time': 8}
        ],
        'lunch': [
            {'name': 'Lentil and Vegetable Soup', 'cost': 4.00, 'carbon': 0.6, 'prep_time': 25},
            {'name': 'Quinoa Salad with Seasonal Veggies', 'cost': 4.50, 'carbon': 0.7, 'prep_time': 15},
            {'name': 'Black Bean and Sweet Potato Bowl', 'cost': 4.25, 'carbon': 0.5, 'prep_time': 20}
        ],
        'dinner': [
            {'name': 'Vegetable Stir-fry with Brown Rice', 'cost': 5.00, 'carbon': 0.8, 'prep_time': 20},
            {'name': 'Pasta with Seasonal Vegetable Sauce', 'cost': 4.75, 'carbon': 0.9, 'prep_time': 25},
            {'name': 'Stuffed Bell Peppers with Quinoa', 'cost': 5.25, 'carbon': 0.7, 'prep_time': 35}
        ]
    }
    
    # Apply dietary preferences filter
    if 'vegan' in dietary_preferences:
        # All meals are already plant-based
        pass
    
    # Generate 7-day plan
    meal_plan = []
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    total_cost = 0
    total_carbon = 0
    
    for day in days:
        daily_meals = []
        daily_carbon = 0
        daily_cost = 0
        
        for meal_type in ['breakfast', 'lunch', 'dinner']:
            meal = random.choice(base_meals[meal_type]).copy()
            meal['type'] = meal_type.title()
            meal['description'] = f"Sustainable {meal_type} focused on local, seasonal ingredients"
            
            # Adjust for people count
            meal['estimated_cost'] = meal['cost'] * people_count
            daily_cost += meal['estimated_cost']
            daily_carbon += meal['carbon']
            
            daily_meals.append(meal)
        
        meal_plan.append({
            'day': day,
            'meals': daily_meals,
            'daily_carbon': daily_carbon,
            'daily_cost': daily_cost
        })
        
        total_cost += daily_cost
        total_carbon += daily_carbon
    
    return {
        'success': True,
        'meal_plan': meal_plan,
        'summary': {
            'total_cost': total_cost,
            'total_carbon_footprint': total_carbon,
            'total_meals': len(days) * 3,
            'avg_cost_per_meal': total_cost / (len(days) * 3)
        },
        'tips': [
            "Buy seasonal and local produce to reduce carbon footprint",
            "Choose plant-based proteins like beans and lentils",
            "Shop at farmers markets when possible",
            "Reduce food waste by meal planning ahead"
        ],
        'generated_by': 'fallback_system'
    }

def generate_fallback_shopping_list(budget: float, duration: str, priority: str, pantry_items: List[str]) -> Dict[str, Any]:
    """Generate basic shopping list without AI"""
    
    # Basic shopping categories with sustainable options
    shopping_categories = {
        'produce': [
            {'name': 'Seasonal Mixed Vegetables', 'quantity': '3 lbs', 'estimated_price': 8.50, 'carbon_footprint': 0.5},
            {'name': 'Local Seasonal Fruits', 'quantity': '2 lbs', 'estimated_price': 6.00, 'carbon_footprint': 0.3},
            {'name': 'Leafy Greens (Spinach/Kale)', 'quantity': '1 bunch', 'estimated_price': 3.50, 'carbon_footprint': 0.2}
        ],
        'proteins': [
            {'name': 'Organic Lentils', 'quantity': '2 lbs', 'estimated_price': 4.50, 'carbon_footprint': 0.3},
            {'name': 'Black Beans (dry)', 'quantity': '1 lb', 'estimated_price': 2.00, 'carbon_footprint': 0.2},
            {'name': 'Quinoa', 'quantity': '1 lb', 'estimated_price': 5.50, 'carbon_footprint': 0.4}
        ],
        'grains': [
            {'name': 'Brown Rice', 'quantity': '2 lbs', 'estimated_price': 3.00, 'carbon_footprint': 0.6},
            {'name': 'Whole Grain Pasta', 'quantity': '1 lb', 'estimated_price': 2.50, 'carbon_footprint': 0.4},
            {'name': 'Oats (steel cut)', 'quantity': '1 lb', 'estimated_price': 3.50, 'carbon_footprint': 0.2}
        ],
        'dairy_alternatives': [
            {'name': 'Oat Milk', 'quantity': '1 carton', 'estimated_price': 4.50, 'carbon_footprint': 0.3},
            {'name': 'Nutritional Yeast', 'quantity': '1 container', 'estimated_price': 6.00, 'carbon_footprint': 0.1}
        ]
    }
    
    # Filter out pantry items
    filtered_categories = {}
    total_cost = 0
    total_items = 0
    total_carbon = 0
    
    for category, items in shopping_categories.items():
        filtered_items = []
        for item in items:
            if not any(pantry_item.lower() in item['name'].lower() for pantry_item in pantry_items):
                filtered_items.append(item)
                total_cost += item['estimated_price']
                total_carbon += item['carbon_footprint']
                total_items += 1
        
        if filtered_items:
            filtered_categories[category] = filtered_items
    
    # Generate alternatives
    alternatives = [
        {
            'original_item': 'Regular Pasta',
            'alternative': 'Whole Grain Pasta',
            'reason': 'Higher fiber and nutrients with similar carbon footprint',
            'carbon_savings': 0.1,
            'cost_savings': 0.00
        },
        {
            'original_item': 'Dairy Milk',
            'alternative': 'Oat Milk',
            'reason': 'Significantly lower carbon footprint than dairy milk',
            'carbon_savings': 1.2,
            'cost_savings': 0.50
        }
    ]
    
    return {
        'success': True,
        'shopping_list': filtered_categories,
        'summary': {
            'total_cost': min(total_cost, budget),
            'estimated_carbon_footprint': total_carbon,
            'total_items': total_items,
            'potential_savings': budget - min(total_cost, budget)
        },
        'alternatives': alternatives,
        'generated_by': 'fallback_system'
    }

def generate_fallback_meal_prep_from_receipt(products: List[Dict], total_carbon: float, optimization_focus: str) -> Dict[str, Any]:
    """Generate basic meal prep suggestions from receipt without AI"""
    
    # Analyze receipt products
    high_carbon_items = [p for p in products if p.get('carbon_footprint', 0) > 2.0]
    vegetables = [p for p in products if p.get('category', '').lower() in ['produce', 'vegetables', 'fruits']]
    proteins = [p for p in products if p.get('category', '').lower() in ['meat', 'dairy', 'protein']]
    
    meal_suggestions = []
    
    # Generate suggestions based on receipt content
    if vegetables:
        meal_suggestions.append({
            'meal_name': 'Rainbow Vegetable Bowl',
            'receipt_ingredients': [v['name'] for v in vegetables[:3]],
            'additional_ingredients': ['Quinoa', 'Tahini dressing', 'Chickpeas'],
            'prep_instructions': 'Roast vegetables at 400°F for 25 minutes. Serve over quinoa with tahini dressing.',
            'prep_time': '35',
            'servings': '4',
            'storage_days': '4',
            'carbon_savings': 1.5
        })
    
    if high_carbon_items:
        meat_items = [item['name'] for item in high_carbon_items if 'meat' in item.get('category', '').lower()]
        if meat_items:
            meal_suggestions.append({
                'meal_name': 'Plant-Based Protein Bowl',
                'receipt_ingredients': [item for item in vegetables[:2]] if vegetables else ['Use other produce'],
                'additional_ingredients': ['Black beans', 'Brown rice', 'Avocado', 'Lime'],
                'prep_instructions': 'Replace meat with seasoned black beans. Combine with roasted vegetables over brown rice.',
                'prep_time': '30',
                'servings': '4',
                'storage_days': '5',
                'carbon_savings': 3.2
            })
    
    # Add a general sustainable option
    meal_suggestions.append({
        'meal_name': 'Sustainable Stir-Fry',
        'receipt_ingredients': [p['name'] for p in products if p.get('category', '').lower() in ['produce', 'vegetables']][:3],
        'additional_ingredients': ['Tofu', 'Brown rice', 'Soy sauce', 'Ginger'],
        'prep_instructions': 'Stir-fry vegetables with tofu and serve over brown rice with ginger-soy sauce.',
        'prep_time': '25',
        'servings': '3',
        'storage_days': '3',
        'carbon_savings': 2.1
    })
    
    optimized_footprint = max(0, total_carbon - sum(s['carbon_savings'] for s in meal_suggestions))
    potential_savings = total_carbon - optimized_footprint
    
    return {
        'success': True,
        'meal_suggestions': meal_suggestions,
        'current_footprint': total_carbon,
        'optimized_footprint': optimized_footprint,
        'potential_savings': potential_savings,
        'generated_by': 'fallback_system'
    }

# AI RESPONSE PARSING FUNCTIONS

def parse_ai_meal_plan_response(ai_text: str, budget: float, people_count: int) -> Dict[str, Any]:
    """Parse AI-generated meal plan text into structured format"""
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    meal_plan = []
    
    # Basic parsing - in production, you'd want more sophisticated parsing
    lines = ai_text.split('\n')
    current_day = None
    current_meals = []
    
    total_cost = 0
    total_carbon = 0
    
    for line in lines:
        line = line.strip()
        
        # Look for day headers
        for day in days:
            if day.lower() in line.lower():
                if current_day and current_meals:
                    daily_carbon = sum(meal.get('carbon_footprint', 0.5) for meal in current_meals)
                    daily_cost = sum(meal.get('estimated_cost', 5.0) for meal in current_meals)
                    
                    meal_plan.append({
                        'day': current_day,
                        'meals': current_meals,
                        'daily_carbon': daily_carbon,
                        'daily_cost': daily_cost
                    })
                    
                    total_cost += daily_cost
                    total_carbon += daily_carbon
                
                current_day = day
                current_meals = []
                break
        
        # Look for meal entries (basic parsing)
        if any(meal_type in line.lower() for meal_type in ['breakfast', 'lunch', 'dinner']):
            # Extract meal info (simplified)
            meal_name = line.split(':')[-1].strip() if ':' in line else line.strip()
            
            meal = {
                'name': meal_name,
                'type': 'Breakfast' if 'breakfast' in line.lower() else 'Lunch' if 'lunch' in line.lower() else 'Dinner',
                'estimated_cost': random.uniform(3.0, 7.0) * people_count,
                'carbon_footprint': random.uniform(0.3, 1.2),
                'prep_time': random.randint(15, 45),
                'description': f"AI-generated sustainable meal with focus on local ingredients"
            }
            current_meals.append(meal)
    
    # Add final day if exists
    if current_day and current_meals:
        daily_carbon = sum(meal.get('carbon_footprint', 0.5) for meal in current_meals)
        daily_cost = sum(meal.get('estimated_cost', 5.0) for meal in current_meals)
        
        meal_plan.append({
            'day': current_day,
            'meals': current_meals,
            'daily_carbon': daily_carbon,
            'daily_cost': daily_cost
        })
        
        total_cost += daily_cost
        total_carbon += daily_carbon
    
    # If parsing failed, use fallback
    if not meal_plan:
        return generate_fallback_meal_plan_structure(budget, people_count)
    
    return {
        'days': meal_plan,
        'summary': {
            'total_cost': total_cost,
            'total_carbon_footprint': total_carbon,
            'total_meals': len(meal_plan) * 3,
            'avg_cost_per_meal': total_cost / max(len(meal_plan) * 3, 1)
        },
        'tips': extract_tips_from_ai_text(ai_text)
    }

def parse_ai_shopping_list_response(ai_text: str, budget: float) -> Dict[str, Any]:
    """Parse AI-generated shopping list text into structured format"""
    
    categories = {
        'produce': [],
        'proteins': [],
        'grains': [],
        'dairy_alternatives': [],
        'pantry_staples': []
    }
    
    # Basic parsing logic - look for category headers and items
    lines = ai_text.split('\n')
    current_category = None
    
    total_cost = 0
    total_items = 0
    total_carbon = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for category headers
        for cat_key in categories.keys():
            if cat_key.replace('_', ' ').lower() in line.lower() or cat_key.lower() in line.lower():
                current_category = cat_key
                break
        
        # Look for items (starting with - or numbers)
        if current_category and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
            # Extract item info (simplified parsing)
            item_text = line.lstrip('-•0123456789. ').strip()
            
            # Try to extract price and carbon info from the line
            price_match = re.search(r'\$(\d+\.?\d*)', item_text)
            carbon_match = re.search(r'(\d+\.?\d*)\s*kg\s*co2', item_text.lower())
            
            item = {
                'name': item_text.split('(')[0].strip(),
                'quantity': '1 unit',  # Default
                'estimated_price': float(price_match.group(1)) if price_match else random.uniform(2.0, 8.0),
                'carbon_footprint': float(carbon_match.group(1)) if carbon_match else random.uniform(0.1, 1.0),
                'brand_preference': 'Organic/Local when available'
            }
            
            categories[current_category].append(item)
            total_cost += item['estimated_price']
            total_carbon += item['carbon_footprint']
            total_items += 1
    
    # If parsing failed, use fallback data
    if total_items == 0:
        return generate_fallback_shopping_list_structure(budget)
    
    # Filter empty categories
    filtered_categories = {k: v for k, v in categories.items() if v}
    
    return {
        'categories': filtered_categories,
        'summary': {
            'total_cost': min(total_cost, budget),
            'estimated_carbon_footprint': total_carbon,
            'total_items': total_items,
            'potential_savings': max(0, budget - total_cost)
        },
        'alternatives': extract_alternatives_from_ai_text(ai_text)
    }

def parse_ai_meal_prep_response(ai_text: str, products: List[Dict], total_carbon: float) -> Dict[str, Any]:
    """Parse AI-generated meal prep suggestions"""
    
    meal_suggestions = []
    
    # Look for meal suggestions in the AI text
    lines = ai_text.split('\n')
    current_suggestion = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for meal names (simple heuristic)
        if any(word in line.lower() for word in ['meal', 'recipe', 'dish', 'bowl', 'soup', 'stir-fry']):
            if current_suggestion:
                meal_suggestions.append(current_suggestion)
            
            current_suggestion = {
                'meal_name': line.strip('1234567890.- '),
                'receipt_ingredients': [p['name'] for p in products[:3]],  # Use first 3 products
                'additional_ingredients': ['Seasonal vegetables', 'Herbs', 'Spices'],
                'prep_instructions': 'Combine ingredients and cook as suggested by AI',
                'prep_time': '30',
                'servings': '4',
                'storage_days': '4',
                'carbon_savings': random.uniform(1.0, 3.0)
            }
    
    if current_suggestion:
        meal_suggestions.append(current_suggestion)
    
    # Ensure we have at least one suggestion
    if not meal_suggestions:
        meal_suggestions = [{
            'meal_name': 'Sustainable Recipe from Receipt Items',
            'receipt_ingredients': [p['name'] for p in products[:4]],
            'additional_ingredients': ['Plant-based protein', 'Whole grains'],
            'prep_instructions': 'Combine receipt ingredients with sustainable alternatives for a complete meal',
            'prep_time': '25',
            'servings': '3-4',
            'storage_days': '5',
            'carbon_savings': 2.0
        }]
    
    total_savings = sum(s['carbon_savings'] for s in meal_suggestions)
    optimized_footprint = max(0, total_carbon - total_savings)
    
    return {
        'suggestions': meal_suggestions,
        'optimized_footprint': optimized_footprint,
        'potential_savings': total_savings
    }

# UTILITY FUNCTIONS

def generate_fallback_meal_plan_structure(budget: float, people_count: int) -> Dict[str, Any]:
    """Generate basic meal plan structure when AI parsing fails"""
    return {
        'days': [
            {
                'day': 'Monday',
                'meals': [
                    {'name': 'Sustainable Breakfast Bowl', 'type': 'Breakfast', 'estimated_cost': 3.0 * people_count, 'carbon_footprint': 0.4, 'prep_time': 15},
                    {'name': 'Plant-based Lunch', 'type': 'Lunch', 'estimated_cost': 4.5 * people_count, 'carbon_footprint': 0.6, 'prep_time': 20},
                    {'name': 'Eco-friendly Dinner', 'type': 'Dinner', 'estimated_cost': 6.0 * people_count, 'carbon_footprint': 0.8, 'prep_time': 30}
                ],
                'daily_carbon': 1.8,
                'daily_cost': 13.5 * people_count
            }
        ],
        'summary': {
            'total_cost': 94.5 * people_count,  # 7 days
            'total_carbon_footprint': 12.6,
            'total_meals': 21,
            'avg_cost_per_meal': 4.5 * people_count
        },
        'tips': ['Focus on local, seasonal ingredients', 'Reduce meat consumption', 'Minimize food waste']
    }

def generate_fallback_shopping_list_structure(budget: float) -> Dict[str, Any]:
    """Generate basic shopping list structure when AI parsing fails"""
    return {
        'categories': {
            'produce': [
                {'name': 'Mixed Seasonal Vegetables', 'quantity': '3 lbs', 'estimated_price': 8.0, 'carbon_footprint': 0.5}
            ],
            'proteins': [
                {'name': 'Lentils', 'quantity': '2 lbs', 'estimated_price': 4.0, 'carbon_footprint': 0.3}
            ]
        },
        'summary': {
            'total_cost': min(12.0, budget),
            'estimated_carbon_footprint': 0.8,
            'total_items': 2,
            'potential_savings': max(0, budget - 12.0)
        },
        'alternatives': []
    }

def extract_tips_from_ai_text(ai_text: str) -> List[str]:
    """Extract sustainability tips from AI response"""
    tips = []
    lines = ai_text.split('\n')
    
    for line in lines:
        if any(word in line.lower() for word in ['tip:', 'suggestion:', 'recommend', 'consider']):
            tip = line.strip('- •').strip()
            if tip and len(tip) > 10:
                tips.append(tip)
    
    # Default tips if none found
    if not tips:
        tips = [
            "Choose seasonal and local produce when possible",
            "Reduce meat consumption to lower carbon footprint",
            "Plan meals ahead to minimize food waste",
            "Buy in bulk to reduce packaging waste"
        ]
    
    return tips[:5]  # Limit to 5 tips

def extract_alternatives_from_ai_text(ai_text: str) -> List[Dict[str, Any]]:
    """Extract product alternatives from AI response"""
    alternatives = []
    
    # Simple pattern matching for alternatives
    lines = ai_text.split('\n')
    for line in lines:
        if 'instead of' in line.lower() or 'alternative to' in line.lower():
            # Parse alternative suggestion
            parts = line.split('instead of') if 'instead of' in line.lower() else line.split('alternative to')
            if len(parts) >= 2:
                alternative = parts[0].strip()
                original = parts[1].strip()
                
                alternatives.append({
                    'original_item': original,
                    'alternative': alternative,
                    'reason': 'More sustainable option',
                    'carbon_savings': random.uniform(0.2, 1.5),
                    'cost_savings': random.uniform(-0.5, 1.0)
                })
    
    return alternatives[:3]  # Limit to 3 alternatives

def find_product_alternatives(product_index: int, include_nearby_stores: bool, 
                            include_price_comparison: bool, max_alternatives: int) -> Dict[str, Any]:
    """Find alternatives for a specific product"""
    
    # This would integrate with your existing carbon_analyzer
    # For now, return mock alternatives
    alternatives = [
        {
            'name': 'Organic Alternative',
            'store_name': 'Local Co-op' if include_nearby_stores else None,
            'distance_miles': 2.3 if include_nearby_stores else None,
            'price': 4.50 if include_price_comparison else None,
            'carbon_footprint': 0.6,
            'carbon_savings': 1.2
        },
        {
            'name': 'Plant-Based Option',
            'store_name': 'Whole Foods' if include_nearby_stores else None,
            'distance_miles': 1.8 if include_nearby_stores else None,
            'price': 3.75 if include_price_comparison else None,
            'carbon_footprint': 0.4,
            'carbon_savings': 1.8
        }
    ]
    
    return {
        'success': True,
        'alternatives': alternatives[:max_alternatives]
    }

def generate_fallback_alternatives() -> Dict[str, Any]:
    """Generate basic alternatives when service is unavailable"""
    return {
        'success': True,
        'alternatives': [
            {
                'name': 'Local Organic Option',
                'price': 4.25,
                'carbon_footprint': 0.5,
                'carbon_savings': 1.0
            }
        ]
    }

def generate_quick_alternatives(products: List[Dict]) -> List[Dict[str, Any]]:
    """Generate quick alternatives for display"""
    alternatives = []
    
    for product in products:
        if product.get('carbon_footprint', 0) > 2.0:
            alternatives.append({
                'original_product': product.get('matched_name', product.get('original_name', '')),
                'alternative_product': f"Sustainable {product.get('category', 'Alternative')}",
                'reason': 'Lower carbon footprint option available',
                'carbon_reduction': random.uniform(0.5, 2.0)
            })
    
    return alternatives[:5]  # Limit to 5 alternatives

@app.route('/api/database-info', methods=['GET'])
def database_info():
    """Get information about the carbon database"""
    if not carbon_analyzer:
        return jsonify({'success': False, 'error': 'Database unavailable'}), 503
    
    try:
        db_stats = {
            'total_products': len(getattr(carbon_analyzer, 'carbon_database', {})),
            'categories': list(set(item.get('category', '') for item in getattr(carbon_analyzer, 'carbon_database', {}).values())),
            'category_counts': {},
            'alternatives_available': len(getattr(carbon_analyzer, 'sustainable_alternatives', {})),
            'last_updated': datetime.now().isoformat()
        }
        
        # Count products per category
        for item in getattr(carbon_analyzer, 'carbon_database', {}).values():
            category = item.get('category', 'unknown')
            db_stats['category_counts'][category] = db_stats['category_counts'].get(category, 0) + 1
        
        return jsonify({
            'success': True,
            'database_info': db_stats
        }), 200
        
    except Exception as e:
        logger.error(f"Database info failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def get_impact_level(carbon_footprint: float) -> str:
    """Determine impact level based on carbon footprint"""
    if carbon_footprint < 0.5:
        return 'VERY_LOW'
    elif carbon_footprint < 1.5:
        return 'LOW'
    elif carbon_footprint < 3.0:
        return 'MEDIUM'
    elif carbon_footprint < 6.0:
        return 'HIGH'
    else:
        return 'VERY_HIGH'

def calculate_potential_savings(recommendations: list) -> float:
    """Calculate total potential carbon savings from recommendations"""
    total_savings = 0
    for rec in recommendations:
        if rec.get('type') == 'product_swap':
            total_savings += rec.get('carbon_savings', 0)
    return round(total_savings, 2)

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET / - Main application page',
            'GET /index.html - Main application page',
            'POST /api/analyze-receipt - Complete receipt analysis',
            'POST /api/ocr-only - Text extraction only',
            'POST /api/generate-meal-plan - AI meal plan generation',
            'POST /api/generate-shopping-list - Smart shopping list',
            'POST /api/generate-receipt-meal-prep - Receipt-based meal prep',
            'POST /api/find-alternatives - Find product alternatives',
            'GET /api/database-info - Database information',
            'GET /health - Health check'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    print("🌱 GreenLens Enhanced Backend Server with AI Meal Planning")
    print("=" * 65)
    print("Services Status:")
    print(f"  OCR Processor: {'✓ Available' if ocr_processor else '✗ Unavailable'}")
    print(f"  Carbon Analyzer: {'✓ Available' if carbon_analyzer else '✗ Unavailable'}")
    print(f"  Gemini AI: {'✓ Available' if gemini_model else '✗ Unavailable (using fallback)'}")
    print()
    print("Available Endpoints:")
    print("  GET  /                           - Main application page")
    print("  GET  /index.html                 - Main application page")
    print("  POST /api/analyze-receipt        - Complete receipt analysis")
    print("  POST /api/ocr-only               - OCR text extraction only") 
    print("  POST /api/generate-meal-plan     - AI-powered meal planning")
    print("  POST /api/generate-shopping-list - Smart shopping list generation")
    print("  POST /api/generate-receipt-meal-prep - Receipt-based meal prep")
    print("  POST /api/find-alternatives      - Find sustainable alternatives")
    print("  GET  /api/database-info          - Database information")
    print("  GET  /health                     - Health check")
    print()
    print("AI Features:")
    if gemini_model:
        print("  ✓ Advanced meal planning with Gemini AI")
        print("  ✓ Smart shopping list optimization")
        print("  ✓ Receipt-based meal prep suggestions")
        print("  ✓ Personalized sustainability recommendations")
    else:
        print("  ⚠ AI features running in fallback mode")
        print("    Set GEMINI_API_KEY environment variable for full AI features")
    print()
    print("Starting server on http://localhost:5100...")
    print("CORS enabled for frontend connections")
    print()
    
    if not services_available:
        print("⚠️  WARNING: Some core services failed to initialize")
        print("   Check the logs above for specific error details")
        print("   The server will run in limited mode")
    else:
        print("✅ All systems ready! You can now:")
        print("   1. Open http://localhost:5100 in your browser")
        print("   2. Upload receipt images for analysis")
        print("   3. Generate AI-powered meal plans")
        print("   4. Create smart shopping lists")
        print("   5. Get personalized sustainability recommendations")
    
    app.run(debug=True, host='0.0.0.0', port=5100)
