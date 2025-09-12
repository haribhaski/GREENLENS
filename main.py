from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import json
import logging
from datetime import datetime
import traceback
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize components with better error handling
ocr_processor = None
carbon_analyzer = None

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
        },
        'version': '2.0.0',
        'features': [
            'Advanced OCR with dual-engine processing',
            'Comprehensive carbon footprint analysis',
            'Sustainable alternatives recommendation',
            'Real-time analytics and caching'
        ] if services_available else ['Limited functionality - services not available'],
        'available_endpoints': [
            'GET / - Main application page',
            'GET /index.html - Main application page',
            'POST /api/scan-receipt - Complete receipt analysis',
            'POST /api/ocr-only - Text extraction only',
            'POST /api/carbon-analysis - Carbon footprint analysis only',
            'GET /api/analytics - Usage analytics',
            'GET /api/database-info - Database information',
            'GET /health - Health check'
        ]
    }
    
    if services_available:
        return jsonify(status), 200
    else:
        return jsonify(status), 503

@app.route('/api/scan-receipt', methods=['POST'])
def scan_receipt_full():
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
        
        # Validate request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided',
                'code': 'MISSING_IMAGE'
            }), 400
        
        logger.info("Starting complete receipt analysis pipeline")
        
        # Step 1: OCR Processing
        logger.info("Phase 1: OCR Text Extraction")
        ocr_result = ocr_processor.process_receipt(data['image'])
        
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
        
        if not products:
            return jsonify({
                'success': False,
                'error': 'No products found in receipt',
                'ocr_result': ocr_result,
                'code': 'NO_PRODUCTS_FOUND'
            }), 422
        
        # Step 3: Carbon Footprint Analysis
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
            
            # OCR Results
            'ocr_data': {
                'raw_text': ocr_result.get('raw_text', ''),
                'confidence': ocr_result.get('processing_info', {}).get('confidence', 0),
                'primary_engine': ocr_result.get('processing_info', {}).get('primary_ocr', 'unknown')
            },
            
            # Structured Receipt Data
            'receipt_info': {
                'store_name': store_info['store_name'],
                'date': store_info['date'],
                'total_amount': store_info['total'],
                'tax': ocr_result.get('structured_data', {}).get('tax', 0),
                'payment_method': ocr_result.get('structured_data', {}).get('payment_method', ''),
                'product_count': len(products)
            },
            
            # Carbon Analysis Results
            'carbon_analysis': {
                'total_carbon_footprint': round(carbon_result.get('total_carbon', 0), 2),
                'average_carbon_per_product': round(carbon_result.get('statistics', {}).get('avg_carbon_per_product', 0), 2),
                'environmental_score': carbon_result.get('environmental_score', 'N/A'),
                'match_rate': round(carbon_result.get('statistics', {}).get('match_rate', 0), 1)
            },
            
            # Detailed Product Analysis
            'products': [
                {
                    'original_name': p.get('original_name', ''),
                    'matched_name': p.get('matched_name', ''),
                    'price': p.get('price', 0),
                    'carbon_footprint': round(p.get('carbon_footprint', 0), 2),
                    'carbon_per_dollar': round(p.get('carbon_per_dollar', 0), 2) if p.get('carbon_per_dollar', 0) > 0 else 0,
                    'category': p.get('category', 'unknown'),
                    'confidence': round(p.get('match_confidence', 0) * 100, 1),
                    'alternatives': p.get('alternatives', [])[:2] if p.get('alternatives') else [],
                    'impact_level': get_impact_level(p.get('carbon_footprint', 0)),
                    'modifiers': p.get('modifiers', [])
                }
                for p in carbon_result.get('matched_products', [])
            ],
            
            # Unmatched Products
            'unmatched_products': [
                {
                    'name': p.get('original_name', ''),
                    'price': p.get('price', 0),
                    'reason': 'No matching product found in database'
                }
                for p in carbon_result.get('unmatched_products', [])
            ],
            
            # Sustainability Recommendations
            'recommendations': carbon_result.get('recommendations', [])[:5],
            
            # Category Breakdown
            'category_breakdown': carbon_result.get('statistics', {}).get('category_breakdown', {}),
            
            # Summary Statistics
            'summary': {
                'products_analyzed': len(carbon_result.get('matched_products', [])),
                'total_products': len(products),
                'highest_impact_product': carbon_result.get('statistics', {}).get('highest_carbon_product', ''),
                'lowest_impact_product': carbon_result.get('statistics', {}).get('lowest_carbon_product', ''),
                'potential_savings': calculate_potential_savings(carbon_result.get('recommendations', []))
            }
        }
        
        logger.info(f"Analysis complete - Total CO2: {complete_result['carbon_analysis']['total_carbon_footprint']}kg")
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
            'POST /api/scan-receipt - Complete receipt analysis',
            'POST /api/ocr-only - Text extraction only',
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
    print("🌱 GreenLens Integrated Backend Server")
    print("=" * 50)
    print("Services Status:")
    print(f"  OCR Processor: {'✓ Available' if ocr_processor else '✗ Unavailable'}")
    print(f"  Carbon Analyzer: {'✓ Available' if carbon_analyzer else '✗ Unavailable'}")
    print()
    print("Available Endpoints:")
    print("  GET  /                      - Main application page")
    print("  GET  /index.html            - Main application page")
    print("  POST /api/scan-receipt      - Complete analysis pipeline")
    print("  POST /api/ocr-only          - OCR text extraction only") 
    print("  GET  /api/database-info     - Database information")
    print("  GET  /health                - Health check")
    print()
    print("Starting server on http://localhost:5100...")
    print("CORS enabled for frontend connections")
    print()
    
    if not services_available:
        print("⚠️  WARNING: Some services failed to initialize")
        print("   Check the logs above for specific error details")
        print("   The server will run in limited mode")
    else:
        print("✅ All systems ready! You can now:")
        print("   1. Open http://localhost:5100 in your browser")
        print("   2. Upload receipt images for analysis")
        print("   3. View carbon footprint results")
    
    app.run(debug=True, host='0.0.0.0', port=5100)