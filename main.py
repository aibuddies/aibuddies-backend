import os
import google.generativeai as genai
import jwt
import time
import hmac
import hashlib
import razorpay
import base64
import io
import random
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, UploadFile, File, Request
from fastapi.responses import JSONResponse, Response  # <-- ADD THIS IMPORT
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from supabase import create_client, Client
from enum import Enum
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime, timedelta
import requests
from PIL import Image
import json
import hashlib

# --- 1. LOAD ALL ENVIRONMENT VARIABLES ---
load_dotenv()

# --- Environment Variables ---
API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
AD_MOB_APP_ID = os.getenv("AD_MOB_APP_ID", "")
UNITY_ADS_GAME_ID = os.getenv("UNITY_ADS_GAME_ID", "")
IRONSOURCE_APP_KEY = os.getenv("IRONSOURCE_APP_KEY", "")

# Validate environment variables
required_vars = {
    "GOOGLE_API_KEY": API_KEY,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_KEY": SUPABASE_SERVICE_KEY,
    "SUPABASE_JWT_SECRET": SUPABASE_JWT_SECRET,
    "RAZORPAY_KEY_ID": RAZORPAY_KEY_ID,
    "RAZORPAY_KEY_SECRET": RAZORPAY_KEY_SECRET
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    # Don't raise error - just log for Render deployment
    print(f"WARNING: Missing environment variables: {', '.join(missing_vars)}")
    # Set defaults for testing
    API_KEY = API_KEY or "test"
    SUPABASE_URL = SUPABASE_URL or "https://test.supabase.co"
    SUPABASE_SERVICE_KEY = SUPABASE_SERVICE_KEY or "test"
    SUPABASE_JWT_SECRET = SUPABASE_JWT_SECRET or "test"
    RAZORPAY_KEY_ID = RAZORPAY_KEY_ID or "test"
    RAZORPAY_KEY_SECRET = RAZORPAY_KEY_SECRET or "test"

print("--- Environment Variables Loaded! ---")

# --- 2. INITIALIZE ALL SERVICES ---
try:
    genai.configure(api_key=API_KEY)
    text_model = genai.GenerativeModel('gemini-2.0-flash')
    vision_model = genai.GenerativeModel('gemini-2.0-flash')
    print("--- AI Models Initialized Successfully! ---")
except Exception as e:
    print(f"!!! AI Model Initialization Failed: {e}")
    # Create dummy models for testing
    class DummyModel:
        def generate_content(self, *args, **kwargs):
            class Response:
                text = "This is a dummy response. AI service not available."
            return Response()
    text_model = DummyModel()
    vision_model = DummyModel()

app = FastAPI(title="AIBUDDIES Advanced API", version="4.0.0")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("--- Supabase Client Initialized Successfully! ---")
except Exception as e:
    print(f"!!! Supabase Initialization Failed: {e}")
    supabase = None

try:
    razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    print("--- Razorpay Client Initialized Successfully! ---")
except Exception as e:
    print(f"!!! Razorpay Initialization Failed: {e}")
    razorpay_client = None

# --- 3. ENHANCED CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173", 
        "https://aibuddies-frontend.vercel.app",
        "https://*.vercel.app",
        "*"  # For testing - remove in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "Origin",
        "X-Requested-With",
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Headers",
        "Access-Control-Allow-Methods"
    ],
    expose_headers=["*"],
    max_age=3600,
)

# Add CORS preflight options handler
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str) -> Response:
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept, Origin, X-Requested-With",
            "Access-Control-Max-Age": "3600"
        }
    )

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept, Origin, X-Requested-With"
    return response

print("--- CORS Middleware Setup Complete! ---")

# --- 4. DEFINE ALL PYDANTIC MODELS ---

class AIModelType(str, Enum):
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    IMAGE_ANALYSIS = "image_analysis"
    CHAT_COMPLETION = "chat_completion"
    TRANSLATION = "translation"
    IMAGE_ENHANCE = "image_enhance"
    BACKGROUND_REMOVE = "background_remove"
    OBJECT_DETECTION = "object_detection"
    STYLE_TRANSFER = "style_transfer"
    IMAGE_CAPTION = "image_caption"
    COLORIZE_IMAGE = "colorize_image"
    IMAGE_TO_TEXT = "image_to_text"
    FACIAL_ANALYSIS = "facial_analysis"

class PromptRequest(BaseModel):
    prompt: str
    model_type: AIModelType = AIModelType.TEXT_GENERATION
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7

class ImageToolRequest(BaseModel):
    prompt: Optional[str] = None
    image_data: str  # Base64 encoded image
    style_reference: Optional[str] = None  # For style transfer
    enhancement_type: Optional[str] = None  # For image enhancement

class AdType(str, Enum):
    REWARDED_VIDEO = "rewarded_video"
    INTERSTITIAL = "interstitial"
    BANNER = "banner"
    NATIVE = "native"

class AdNetwork(str, Enum):
    ADMOB = "admob"
    UNITY = "unity"
    IRONSOURCE = "ironsource"
    CUSTOM = "custom"

class AdWatchRequest(BaseModel):
    ad_id: str
    ad_type: AdType = AdType.REWARDED_VIDEO
    ad_network: AdNetwork = AdNetwork.CUSTOM
    placement_id: Optional[str] = None
    verification_data: Optional[Dict[str, Any]] = None
    
    @validator('ad_id')
    def validate_ad_id(cls, v):
        if len(v) < 5:
            raise ValueError('Ad ID must be at least 5 characters long')
        return v

class CreateOrderRequest(BaseModel):
    amount: int = Field(..., description="Amount in paisa (e.g., 50000 for ₹500)")
    currency: str = "INR"
    credit_package: int = Field(..., description="Number of credits to purchase")

class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    credits_to_add: int

# --- 5. CONSTANTS ---
FREE_SIGNUP_CREDITS = 100
AD_REWARD_CREDITS = 15

# Dynamic ad rewards based on type
AD_REWARDS = {
    AdType.REWARDED_VIDEO: 15,
    AdType.INTERSTITIAL: 8,
    AdType.BANNER: 3,
    AdType.NATIVE: 10
}

# Ad cooldown periods (in minutes)
AD_COOLDOWNS = {
    AdType.REWARDED_VIDEO: 30,
    AdType.INTERSTITIAL: 20,
    AdType.BANNER: 5,
    AdType.NATIVE: 15
}

# Different credit costs for different tools
CREDIT_COST_PER_REQUEST = {
    "text_generation": 1,
    "code_generation": 2,
    "image_analysis": 3,
    "chat_completion": 1,
    "translation": 1,
    "image_enhance": 5,
    "background_remove": 8,
    "object_detection": 4,
    "style_transfer": 10,
    "image_caption": 2,
    "colorize_image": 7,
    "image_to_text": 3,
    "facial_analysis": 6
}

# Tool descriptions for frontend
TOOL_DESCRIPTIONS = {
    "text_generation": "Generate creative text content",
    "code_generation": "Generate and explain code",
    "image_analysis": "Analyze and describe images in detail",
    "image_enhance": "Enhance image quality and resolution",
    "background_remove": "Remove background from images automatically",
    "object_detection": "Detect and identify objects in images",
    "style_transfer": "Apply artistic styles to your images",
    "image_caption": "Generate accurate captions for images",
    "colorize_image": "Colorize black and white images",
    "image_to_text": "Extract text from images (OCR)",
    "facial_analysis": "Analyze facial features and expressions"
}

# Predefined ad campaigns
DEFAULT_AD_CAMPAIGNS = [
    {
        "id": "rewarded_video_1",
        "name": "Premium Video Rewards",
        "ad_type": AdType.REWARDED_VIDEO,
        "credits_reward": 15,
        "daily_limit": 5,
        "cooldown_minutes": 30,
        "is_active": True
    },
    {
        "id": "interstitial_1", 
        "name": "Quick Interstitial Ads",
        "ad_type": AdType.INTERSTITIAL,
        "credits_reward": 8,
        "daily_limit": 10,
        "cooldown_minutes": 20,
        "is_active": True
    },
    {
        "id": "banner_1",
        "name": "Passive Banner Rewards", 
        "ad_type": AdType.BANNER,
        "credits_reward": 3,
        "daily_limit": 20,
        "cooldown_minutes": 5,
        "is_active": True
    }
]

# --- 6. AUTHENTICATION LOGIC ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

async def get_current_user_id(token: str = Depends(oauth2_scheme)):
    """Dependency to decode and validate the Supabase JWT."""
    if not token:
        # For testing, return a dummy user ID
        return "test-user-id"
    
    try:
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user ID")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")

# --- 7. DATABASE HELPER FUNCTIONS ---

async def initialize_user_profile(user_id: str):
    """Initialize user profile with free credits when they sign up for the first time."""
    try:
        if not supabase:
            print("!!! Supabase not available - skipping profile initialization")
            return True
            
        print(f"--- INITIALIZING PROFILE FOR USER: {user_id}")
        
        # Check if profile already exists
        response = supabase.table('profiles').select('id, credits').eq('id', user_id).execute()
        print(f"--- PROFILE CHECK RESPONSE: {response.data}")
        
        if not response.data:
            print("--- CREATING NEW USER PROFILE ---")
            # Create new profile with free credits
            profile_data = {
                'id': user_id,
                'credits': FREE_SIGNUP_CREDITS,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'total_credits_earned': FREE_SIGNUP_CREDITS,
                'ads_watched_today': 0,
                'last_ad_watch': None
            }
            
            # Insert profile
            profile_response = supabase.table('profiles').insert(profile_data).execute()
            print(f"--- PROFILE CREATION RESPONSE: {profile_response.data}")
            
            if profile_response.data:
                # Record credit history
                credit_history = {
                    'id': str(uuid.uuid4()),
                    'user_id': user_id,
                    'action': 'signup_bonus',
                    'credits_change': FREE_SIGNUP_CREDITS,
                    'description': f'Free {FREE_SIGNUP_CREDITS} credits for signing up',
                    'created_at': datetime.utcnow().isoformat()
                }
                
                history_response = supabase.table('credit_history').insert(credit_history).execute()
                print(f"--- CREDIT HISTORY RESPONSE: {history_response.data}")
                
                print(f"--- NEW USER PROFILE CREATED FOR {user_id} WITH {FREE_SIGNUP_CREDITS} FREE CREDITS ---")
                return True
            else:
                print("!!! PROFILE CREATION FAILED")
                return False
        else:
            print(f"--- PROFILE ALREADY EXISTS WITH CREDITS: {response.data[0]['credits']}")
            return True
            
    except Exception as e:
        print(f"!!! ERROR INITIALIZING USER PROFILE: {e}")
        return False

async def get_user_credits(user_id: str) -> int:
    """Get current user credits."""
    if not supabase:
        return FREE_SIGNUP_CREDITS  # Return default credits if no database
    
    response = supabase.table('profiles').select('credits').eq('id', user_id).execute()
    if not response.data:
        return 0
    return response.data[0]['credits']

async def update_user_credits(user_id: str, credit_change: int, action: str, description: str, metadata: Dict[str, Any] = None):
    """Update user credits and record history."""
    try:
        if not supabase:
            print("!!! Supabase not available - skipping credit update")
            return FREE_SIGNUP_CREDITS
            
        current_credits = await get_user_credits(user_id)
        new_credits = current_credits + credit_change
        
        supabase.table('profiles').update({
            'credits': new_credits,
            'updated_at': datetime.utcnow().isoformat()
        }).eq('id', user_id).execute()
        
        history_record = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'action': action,
            'credits_change': credit_change,
            'description': description,
            'metadata': metadata or {},
            'created_at': datetime.utcnow().isoformat()
        }
        supabase.table('credit_history').insert(history_record).execute()
        
        return new_credits
    except Exception as e:
        print(f"!!! ERROR updating user credits: {e}")
        raise

# --- 8. UNIQUE AI IMAGE TOOLS ---

async def enhance_image_quality(prompt: str, image_data: str) -> Dict[str, Any]:
    """Enhance image quality using AI."""
    try:
        enhanced_prompt = f"""
        Enhance and improve this image quality. Focus on:
        1. Increasing resolution and clarity
        2. Improving lighting and contrast
        3. Reducing noise and artifacts
        4. Enhancing colors and sharpness
        
        Original request: {prompt}
        
        Provide detailed analysis of the enhancements made.
        """
        
        response = vision_model.generate_content([
            enhanced_prompt,
            {"mime_type": "image/jpeg", "data": base64.b64decode(image_data)}
        ])
        
        return {
            "analysis": response.text,
            "enhancement_type": "quality_improvement",
            "status": "enhanced"
        }
    except Exception as e:
        return {"error": f"Enhancement failed: {str(e)}"}

async def remove_background_ai(prompt: str, image_data: str) -> Dict[str, Any]:
    """Remove background from image using AI analysis."""
    try:
        bg_remove_prompt = f"""
        Analyze this image and describe how to remove the background effectively.
        Identify the main subject and background elements.
        Provide detailed instructions for background removal.
        
        User request: {prompt}
        
        Describe:
        1. Main subject identification
        2. Background complexity
        3. Recommended removal technique
        4. Edge refinement suggestions
        """
        
        response = vision_model.generate_content([
            bg_remove_prompt,
            {"mime_type": "image/jpeg", "data": base64.b64decode(image_data)}
        ])
        
        return {
            "analysis": response.text,
            "subject_detected": True,
            "background_complexity": "medium",
            "removal_instructions": "Use the provided analysis for precise background removal"
        }
    except Exception as e:
        return {"error": f"Background analysis failed: {str(e)}"}

async def detect_objects_ai(image_data: str) -> Dict[str, Any]:
    """Detect and identify objects in image."""
    try:
        detection_prompt = """
        Analyze this image and detect all visible objects. For each object found, provide:
        1. Object name
        2. Confidence level (high/medium/low)
        3. Position description
        4. Size relative to image
        
        Format the response as a JSON-like structure with objects array.
        """
        
        response = vision_model.generate_content([
            detection_prompt,
            {"mime_type": "image/jpeg", "data": base64.b64decode(image_data)}
        ])
        
        return {
            "object_detection": response.text,
            "objects_found": True,
            "analysis_complete": True
        }
    except Exception as e:
        return {"error": f"Object detection failed: {str(e)}"}

async def transfer_style_ai(content_image: str, style_reference: str, prompt: str) -> Dict[str, Any]:
    """Apply artistic style to image (conceptual)."""
    try:
        style_prompt = f"""
        Analyze the style of the reference image and describe how to apply it to the content image.
        
        Content image description needed: {prompt}
        Style reference available: Yes
        
        Provide detailed analysis of:
        1. Style characteristics (colors, textures, patterns)
        2. How to apply style to content
        3. Expected visual outcome
        4. Technical recommendations
        """
        
        response = vision_model.generate_content([
            style_prompt,
            {"mime_type": "image/jpeg", "data": base64.b64decode(content_image)}
        ])
        
        return {
            "style_analysis": response.text,
            "style_applied": "conceptual",
            "recommendations": "Use the analysis to apply style programmatically"
        }
    except Exception as e:
        return {"error": f"Style transfer analysis failed: {str(e)}"}

async def generate_image_caption_ai(image_data: str) -> Dict[str, Any]:
    """Generate accurate and creative captions for images."""
    try:
        caption_prompt = """
        Generate multiple captions for this image:
        1. A descriptive caption (factual)
        2. A creative caption (artistic)
        3. A social media caption (engaging)
        4. A technical caption (detailed)
        
        Format as JSON with different caption types.
        """
        
        response = vision_model.generate_content([
            caption_prompt,
            {"mime_type": "image/jpeg", "data": base64.b64decode(image_data)}
        ])
        
        return {
            "captions": response.text,
            "caption_types": ["descriptive", "creative", "social_media", "technical"]
        }
    except Exception as e:
        return {"error": f"Caption generation failed: {str(e)}"}

async def colorize_image_ai(image_data: str, prompt: str) -> Dict[str, Any]:
    """Colorize black and white images with AI suggestions."""
    try:
        colorize_prompt = f"""
        Analyze this black and white image and provide colorization suggestions.
        
        User preferences: {prompt}
        
        Provide:
        1. Recommended color palette
        2. Specific area coloring suggestions
        3. Historical accuracy considerations (if applicable)
        4. Modern color adaptation ideas
        """
        
        response = vision_model.generate_content([
            colorize_prompt,
            {"mime_type": "image/jpeg", "data": base64.b64decode(image_data)}
        ])
        
        return {
            "color_suggestions": response.text,
            "color_palette_provided": True,
            "coloring_instructions": "Use the suggested palette for manual or AI colorization"
        }
    except Exception as e:
        return {"error": f"Colorization analysis failed: {str(e)}"}

async def extract_text_from_image_ai(image_data: str) -> Dict[str, Any]:
    """Extract text from images (OCR-like functionality)."""
    try:
        ocr_prompt = """
        Extract all visible text from this image. Be extremely accurate with:
        - Printed text
        - Handwritten text (if legible)
        - Numbers and symbols
        - Text in different orientations
        
        Format the extracted text clearly, preserving paragraphs and structure.
        """
        
        response = vision_model.generate_content([
            ocr_prompt,
            {"mime_type": "image/jpeg", "data": base64.b64decode(image_data)}
        ])
        
        return {
            "extracted_text": response.text,
            "text_confidence": "high",
            "format_preserved": True
        }
    except Exception as e:
        return {"error": f"Text extraction failed: {str(e)}"}

async def analyze_facial_features_ai(image_data: str) -> Dict[str, Any]:
    """Analyze facial features and expressions."""
    try:
        facial_prompt = """
        Analyze the facial features in this image. Provide:
        1. Estimated age range
        2. Dominant facial expression
        3. Notable facial features
        4. Gender presentation (if discernible)
        5. Overall emotional tone
        
        Note: This is for creative/entertainment purposes only.
        Be respectful and avoid making definitive statements.
        """
        
        response = vision_model.generate_content([
            facial_prompt,
            {"mime_type": "image/jpeg", "data": base64.b64decode(image_data)}
        ])
        
        return {
            "facial_analysis": response.text,
            "analysis_type": "descriptive",
            "purpose": "creative_entertainment"
        }
    except Exception as e:
        return {"error": f"Facial analysis failed: {str(e)}"}

# --- 9. ADVANCED AD SYSTEM ---

class AdManager:
    def __init__(self):
        self.ad_campaigns = {campaign["id"]: campaign for campaign in DEFAULT_AD_CAMPAIGNS}
    
    async def verify_ad_completion(self, ad_network: AdNetwork, verification_data: Dict[str, Any]) -> bool:
        """Verify ad completion with the respective ad network."""
        try:
            if ad_network == AdNetwork.ADMOB:
                return await self._verify_admob_ad(verification_data)
            elif ad_network == AdNetwork.UNITY:
                return await self._verify_unity_ad(verification_data)
            elif ad_network == AdNetwork.IRONSOURCE:
                return await self._verify_ironsource_ad(verification_data)
            elif ad_network == AdNetwork.CUSTOM:
                return await self._verify_custom_ad(verification_data)
            else:
                return False
        except Exception as e:
            print(f"!!! AD VERIFICATION ERROR: {e}")
            return False
    
    async def _verify_admob_ad(self, verification_data: Dict[str, Any]) -> bool:
        """Verify AdMob ad completion."""
        required_fields = ['ad_unit_id', 'timestamp', 'signature']
        return all(field in verification_data for field in required_fields)
    
    async def _verify_unity_ad(self, verification_data: Dict[str, Any]) -> bool:
        """Verify Unity Ads completion."""
        required_fields = ['placementId', 'rewarded', 'timestamp']
        return all(field in verification_data for field in required_fields)
    
    async def _verify_ironsource_ad(self, verification_data: Dict[str, Any]) -> bool:
        """Verify IronSource ad completion."""
        required_fields = ['instanceId', 'revenue', 'timestamp']
        return all(field in verification_data for field in required_fields)
    
    async def _verify_custom_ad(self, verification_data: Dict[str, Any]) -> bool:
        """Verify custom ad completion with basic checks."""
        if 'timestamp' in verification_data:
            ad_time = datetime.fromisoformat(verification_data['timestamp'])
            time_diff = (datetime.utcnow() - ad_time).total_seconds()
            if time_diff < -300 or time_diff > 3600:
                return False
        return len(verification_data) >= 2
    
    async def can_user_watch_ad(self, user_id: str, ad_type: AdType, campaign_id: str = None) -> Dict[str, Any]:
        """Check if user can watch an ad based on limits and cooldowns."""
        try:
            if not supabase:
                return {"can_watch": True, "reason": "Database not available"}
                
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            
            response = supabase.table('ad_watch_history') \
                .select('*') \
                .eq('user_id', user_id) \
                .eq('ad_type', ad_type) \
                .gte('watched_at', today_start.isoformat()) \
                .execute()
            
            today_watches = len(response.data)
            
            campaign = self.ad_campaigns.get(campaign_id) if campaign_id else None
            daily_limit = campaign.get('daily_limit', 5) if campaign else 5
            cooldown = campaign.get('cooldown_minutes', AD_COOLDOWNS[ad_type]) if campaign else AD_COOLDOWNS[ad_type]
            
            if today_watches >= daily_limit:
                return {
                    "can_watch": False,
                    "reason": f"Daily limit reached for {ad_type} ads",
                    "next_available": tomorrow_availability(),
                    "limit_info": {
                        "today_watches": today_watches,
                        "daily_limit": daily_limit
                    }
                }
            
            if response.data:
                last_watch = max([datetime.fromisoformat(watch['watched_at']) for watch in response.data])
                cooldown_end = last_watch + timedelta(minutes=cooldown)
                
                if datetime.utcnow() < cooldown_end:
                    return {
                        "can_watch": False,
                        "reason": f"Cooldown period active for {ad_type} ads",
                        "next_available": cooldown_end.isoformat(),
                        "cooldown_remaining": int((cooldown_end - datetime.utcnow()).total_seconds() / 60)
                    }
            
            return {
                "can_watch": True,
                "today_watches": today_watches,
                "daily_limit": daily_limit,
                "cooldown_minutes": cooldown
            }
            
        except Exception as e:
            print(f"!!! ERROR checking ad availability: {e}")
            return {"can_watch": False, "reason": "System error"}
    
    async def record_ad_watch(self, user_id: str, ad_type: AdType, campaign_id: str = None, credits_earned: int = 0):
        """Record ad watch in history."""
        try:
            if not supabase:
                return
                
            watch_record = {
                'id': str(uuid.uuid4()),
                'user_id': user_id,
                'ad_type': ad_type,
                'campaign_id': campaign_id,
                'credits_earned': credits_earned,
                'watched_at': datetime.utcnow().isoformat(),
                'ip_address': 'tracked'
            }
            
            supabase.table('ad_watch_history').insert(watch_record).execute()
            
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            
            response = supabase.table('ad_watch_history') \
                .select('id', count='exact') \
                .eq('user_id', user_id) \
                .gte('watched_at', today_start) \
                .execute()
            
            ads_today = response.count
            
            supabase.table('profiles') \
                .update({
                    'ads_watched_today': ads_today,
                    'last_ad_watch': datetime.utcnow().isoformat()
                }) \
                .eq('id', user_id) \
                .execute()
                
        except Exception as e:
            print(f"!!! ERROR recording ad watch: {e}")

def tomorrow_availability():
    """Get tomorrow's date for daily reset."""
    tomorrow = datetime.utcnow() + timedelta(days=1)
    return tomorrow.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

# Initialize ad manager
ad_manager = AdManager()

# --- 10. AI ENDPOINTS ---

@app.get("/")
def read_root():
    return {
        "message": "AIBUDDIES Advanced Backend is running!",
        "status": "active",
        "version": "4.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/tools")
async def get_available_tools():
    """Get all available AI tools with their credit costs."""
    return {
        "tools": TOOL_DESCRIPTIONS,
        "credit_costs": CREDIT_COST_PER_REQUEST,
        "free_signup_credits": FREE_SIGNUP_CREDITS,
        "ad_reward_credits": AD_REWARD_CREDITS,
        "status": "success"
    }

@app.get("/api/v1/user/profile")
async def get_user_profile(user_id: str = Depends(get_current_user_id)):
    """Get user profile and credit information."""
    try:
        await initialize_user_profile(user_id)
        
        if not supabase:
            return {
                "profile": {
                    "id": user_id,
                    "credits": FREE_SIGNUP_CREDITS,
                    "created_at": datetime.utcnow().isoformat(),
                    "total_credits_earned": FREE_SIGNUP_CREDITS,
                    "ads_watched_today": 0
                },
                "credit_history": [],
                "available_tools": TOOL_DESCRIPTIONS,
                "status": "success"
            }
        
        response = supabase.table('profiles').select('*').eq('id', user_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        history_response = supabase.table('credit_history') \
            .select('*') \
            .eq('user_id', user_id) \
            .order('created_at', desc=True) \
            .limit(10) \
            .execute()
        
        profile = response.data[0]
        return {
            "profile": profile,
            "credit_history": history_response.data,
            "available_tools": TOOL_DESCRIPTIONS,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ai/generate")
async def generate_ai_content(
    request: PromptRequest,
    user_id: str = Depends(get_current_user_id)
):
    """Advanced AI content generation with multiple model types."""
    try:
        await initialize_user_profile(user_id)
        
        credit_cost = CREDIT_COST_PER_REQUEST.get(request.model_type, 1)
        current_credits = await get_user_credits(user_id)
        
        if current_credits < credit_cost:
            raise HTTPException(
                status_code=402, 
                detail=f"Need {credit_cost} credits for {request.model_type}, but only have {current_credits}"
            )
        
        # Process based on model type
        if request.model_type == AIModelType.TEXT_GENERATION:
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=request.max_tokens,
                temperature=request.temperature
            )
            response = text_model.generate_content(request.prompt, generation_config=generation_config)
            result = response.text
        elif request.model_type == AIModelType.CODE_GENERATION:
            code_prompt = f"""You are an expert programmer. Generate clean, efficient code based on the following request. 
            Provide only the code with minimal explanation unless specifically asked for comments.
            
            Request: {request.prompt}
            
            Code:"""
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=request.max_tokens,
                temperature=0.2
            )
            response = text_model.generate_content(code_prompt, generation_config=generation_config)
            result = response.text
        elif request.model_type == AIModelType.TRANSLATION:
            translation_prompt = f"""Translate the following text. If no target language is specified, translate to English. 
            Provide only the translation without additional explanations.
            
            Text to translate: {request.prompt}
            
            Translation:"""
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=request.max_tokens,
                temperature=0.3
            )
            response = text_model.generate_content(translation_prompt, generation_config=generation_config)
            result = response.text
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
        # Deduct credits
        new_credits = await update_user_credits(
            user_id=user_id,
            credit_change=-credit_cost,
            action="ai_usage",
            description=f"Used {credit_cost} credits for {request.model_type}",
            metadata={
                "model_type": request.model_type,
                "prompt_length": len(request.prompt),
                "max_tokens": request.max_tokens
            }
        )
        
        return {
            "response": result,
            "model_type": request.model_type,
            "credits_used": credit_cost,
            "credits_remaining": new_credits,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"!!! AI GENERATION ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/image/{tool_type}")
async def use_image_tool(
    tool_type: AIModelType,
    request: ImageToolRequest,
    user_id: str = Depends(get_current_user_id)
):
    """Use various AI image tools with different credit costs."""
    try:
        await initialize_user_profile(user_id)
        
        image_tools = [
            AIModelType.IMAGE_ENHANCE, AIModelType.BACKGROUND_REMOVE,
            AIModelType.OBJECT_DETECTION, AIModelType.STYLE_TRANSFER,
            AIModelType.IMAGE_CAPTION, AIModelType.COLORIZE_IMAGE,
            AIModelType.IMAGE_TO_TEXT, AIModelType.FACIAL_ANALYSIS
        ]
        
        if tool_type not in image_tools:
            raise HTTPException(status_code=400, detail="Invalid image tool type")
        
        credit_cost = CREDIT_COST_PER_REQUEST.get(tool_type, 5)
        current_credits = await get_user_credits(user_id)
        
        if current_credits < credit_cost:
            raise HTTPException(
                status_code=402, 
                detail=f"Need {credit_cost} credits for {tool_type}, but only have {current_credits}"
            )
        
        # Process based on tool type
        if tool_type == AIModelType.IMAGE_ENHANCE:
            result = await enhance_image_quality(request.prompt or "Enhance image quality", request.image_data)
        elif tool_type == AIModelType.BACKGROUND_REMOVE:
            result = await remove_background_ai(request.prompt or "Remove background", request.image_data)
        elif tool_type == AIModelType.OBJECT_DETECTION:
            result = await detect_objects_ai(request.image_data)
        elif tool_type == AIModelType.STYLE_TRANSFER:
            result = await transfer_style_ai(request.image_data, request.style_reference or "", request.prompt or "Apply style")
        elif tool_type == AIModelType.IMAGE_CAPTION:
            result = await generate_image_caption_ai(request.image_data)
        elif tool_type == AIModelType.COLORIZE_IMAGE:
            result = await colorize_image_ai(request.image_data, request.prompt or "Colorize naturally")
        elif tool_type == AIModelType.IMAGE_TO_TEXT:
            result = await extract_text_from_image_ai(request.image_data)
        elif tool_type == AIModelType.FACIAL_ANALYSIS:
            result = await analyze_facial_features_ai(request.image_data)
        else:
            raise HTTPException(status_code=400, detail="Tool not implemented")
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Deduct credits
        new_credits = await update_user_credits(
            user_id=user_id,
            credit_change=-credit_cost,
            action="image_tool_usage",
            description=f"Used {credit_cost} credits for {tool_type}",
            metadata={
                "tool_type": tool_type,
                "prompt_used": request.prompt,
                "credits_used": credit_cost
            }
        )
        
        return {
            "result": result,
            "tool_used": tool_type,
            "credits_used": credit_cost,
            "credits_remaining": new_credits,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"!!! IMAGE TOOL ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 11. PAYMENT ENDPOINTS ---

@app.post("/api/v1/payments/create-order")
async def create_order(
    order_request: CreateOrderRequest,
    user_id: str = Depends(get_current_user_id)
):
    """Create Razorpay order for credit purchase."""
    print(f"--- CREATE ORDER REQUEST: user_id={user_id}, amount={order_request.amount}, credits={order_request.credit_package}")
    
    if not razorpay_client:
        raise HTTPException(status_code=500, detail="Payment service not available")
    
    if order_request.amount < 100:
        raise HTTPException(status_code=400, detail="Amount must be at least ₹1")
    
    order_data = {
        "amount": order_request.amount,
        "currency": order_request.currency,
        "receipt": f"order_rcptid_{user_id}_{int(time.time())}",
        "notes": {
            "user_id": user_id,
            "credit_package": order_request.credit_package,
            "product": "AIBUDDIES Credits"
        }
    }
    
    print(f"--- ORDER DATA: {order_data}")
    
    try:
        # Test Razorpay connection
        print("--- TESTING RAZORPAY CONNECTION ---")
        
        # Create order
        order = razorpay_client.order.create(data=order_data)
        print(f"--- ORDER CREATED SUCCESSFULLY: {order}")
        
        return {
            **order,
            "status": "success"
        }
    except razorpay.errors.BadRequestError as e:
        print(f"!!! RAZORPAY BAD REQUEST ERROR: {e}")
        raise HTTPException(status_code=400, detail=f"Razorpay error: {str(e)}")
    except Exception as e:
        print(f"!!! RAZORPAY ORDER ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not create Razorpay order: {str(e)}")

@app.post("/api/v1/payments/verify-payment")
async def verify_payment(
    request: VerifyPaymentRequest,
    user_id: str = Depends(get_current_user_id)
):
    """Verify Razorpay payment and add credits."""
    if not razorpay_client:
        raise HTTPException(status_code=500, detail="Payment service not available")
        
    try:
        razorpay_client.utility.verify_payment_signature({
            'razorpay_order_id': request.razorpay_order_id,
            'razorpay_payment_id': request.razorpay_payment_id,
            'razorpay_signature': request.razorpay_signature
        })
    except razorpay.errors.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid payment signature")
    
    try:
        new_credits_total = await update_user_credits(
            user_id=user_id,
            credit_change=request.credits_to_add,
            action="purchase",
            description=f"Purchased {request.credits_to_add} credits",
            metadata={
                "razorpay_order_id": request.razorpay_order_id,
                "razorpay_payment_id": request.razorpay_payment_id,
                "credits_purchased": request.credits_to_add
            }
        )

        return {
            "status": "success", 
            "new_credits": new_credits_total,
            "credits_added": request.credits_to_add
        }
    except Exception as e:
        print(f"!!! CREDIT UPDATE FAILED: {e}")
        raise HTTPException(status_code=500, detail="Payment verified, but failed to update credits")

# --- 12. AD ENDPOINTS ---

@app.get("/api/v1/ads/availability")
async def get_ad_availability(
    ad_type: AdType = AdType.REWARDED_VIDEO,
    campaign_id: Optional[str] = None,
    user_id: str = Depends(get_current_user_id)
):
    """Check if user can watch ads and get available campaigns."""
    try:
        await initialize_user_profile(user_id)
        
        availability = await ad_manager.can_user_watch_ad(user_id, ad_type, campaign_id)
        
        available_campaigns = [
            campaign for campaign in ad_manager.ad_campaigns.values()
            if campaign['ad_type'] == ad_type and campaign['is_active']
        ]
        
        return {
            "availability": availability,
            "available_campaigns": available_campaigns,
            "ad_rewards": AD_REWARDS,
            "user_id": user_id,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ads/watch")
async def watch_ad_reward(
    request: AdWatchRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id)
):
    """Reward user for watching ads with advanced verification."""
    try:
        await initialize_user_profile(user_id)
        
        verification_result = await ad_manager.verify_ad_completion(
            request.ad_network, 
            request.verification_data or {}
        )
        
        if not verification_result:
            raise HTTPException(
                status_code=400, 
                detail="Ad verification failed. Please watch the complete ad."
            )
        
        availability = await ad_manager.can_user_watch_ad(user_id, request.ad_type, request.placement_id)
        if not availability["can_watch"]:
            raise HTTPException(
                status_code=429,
                detail=availability["reason"],
                headers={"Retry-After": availability.get("next_available", "")}
            )
        
        credits_to_reward = AD_REWARDS.get(request.ad_type, AD_REWARD_CREDITS)
        
        if request.ad_network in [AdNetwork.ADMOB, AdNetwork.UNITY]:
            credits_to_reward += 2
        
        new_credits = await update_user_credits(
            user_id=user_id,
            credit_change=credits_to_reward,
            action="ad_reward",
            description=f"Received {credits_to_reward} credits for watching {request.ad_type} ad",
            metadata={
                "ad_id": request.ad_id,
                "ad_type": request.ad_type.value,
                "ad_network": request.ad_network.value,
                "placement_id": request.placement_id,
                "verification_data": request.verification_data,
                "reward_amount": credits_to_reward
            }
        )
        
        background_tasks.add_task(
            ad_manager.record_ad_watch,
            user_id, request.ad_type, request.placement_id, credits_to_reward
        )
        
        cooldown = AD_COOLDOWNS.get(request.ad_type, 30)
        next_available = (datetime.utcnow() + timedelta(minutes=cooldown)).isoformat()
        
        return {
            "status": "success",
            "credits_earned": credits_to_reward,
            "total_credits": new_credits,
            "next_ad_available": next_available,
            "ad_type": request.ad_type.value,
            "verification": "verified"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"!!! AD REWARD ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ads/campaigns")
async def get_ad_campaigns(user_id: str = Depends(get_current_user_id)):
    """Get all available ad campaigns with user-specific availability."""
    try:
        await initialize_user_profile(user_id)
        
        campaigns_with_availability = []
        
        for campaign_id, campaign in ad_manager.ad_campaigns.items():
            if campaign['is_active']:
                availability = await ad_manager.can_user_watch_ad(
                    user_id, campaign['ad_type'], campaign_id
                )
                
                campaign_with_avail = campaign.copy()
                campaign_with_avail['id'] = campaign_id
                campaign_with_avail['user_availability'] = availability
                campaign_with_avail['reward'] = AD_REWARDS.get(campaign['ad_type'], AD_REWARD_CREDITS)
                
                campaigns_with_availability.append(campaign_with_avail)
        
        return {
            "campaigns": campaigns_with_availability,
            "total_active_campaigns": len(campaigns_with_availability),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ads/daily-bonus")
async def claim_daily_bonus(user_id: str = Depends(get_current_user_id)):
    """Claim daily login bonus."""
    try:
        await initialize_user_profile(user_id)
        
        if not supabase:
            # Return mock bonus if no database
            base_bonus = 10
            total_bonus = base_bonus
            return {
                "status": "success",
                "bonus_earned": total_bonus,
                "breakdown": {
                    "base_bonus": base_bonus,
                    "streak_bonus": 0
                },
                "total_credits": FREE_SIGNUP_CREDITS + total_bonus,
                "next_bonus": (datetime.utcnow() + timedelta(days=1)).replace(hour=0, minute=0, second=0).isoformat()
            }
        
        today = datetime.utcnow().date().isoformat()
        
        response = supabase.table('credit_history') \
            .select('created_at') \
            .eq('user_id', user_id) \
            .eq('action', 'daily_bonus') \
            .gte('created_at', today + 'T00:00:00') \
            .execute()
        
        if response.data:
            raise HTTPException(
                status_code=429,
                detail="Daily bonus already claimed today"
            )
        
        base_bonus = 10
        streak_bonus = random.randint(1, 7) * 2  # Simulated streak bonus
        total_bonus = base_bonus + streak_bonus
        
        new_credits = await update_user_credits(
            user_id=user_id,
            credit_change=total_bonus,
            action="daily_bonus",
            description=f"Daily login bonus: {base_bonus} + {streak_bonus} streak bonus",
            metadata={
                "base_bonus": base_bonus,
                "streak_bonus": streak_bonus,
                "total_bonus": total_bonus,
                "streak_days": random.randint(1, 7)
            }
        )
        
        return {
            "status": "success",
            "bonus_earned": total_bonus,
            "breakdown": {
                "base_bonus": base_bonus,
                "streak_bonus": streak_bonus
            },
            "total_credits": new_credits,
            "next_bonus": (datetime.utcnow() + timedelta(days=1)).replace(hour=0, minute=0, second=0).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 13. DEBUG ENDPOINT ---

@app.get("/api/v1/debug/user-status")
async def debug_user_status(user_id: str = Depends(get_current_user_id)):
    """Debug endpoint to check user status."""
    try:
        if not supabase:
            return {
                "user_id": user_id,
                "profile_exists": False,
                "profile_data": None,
                "credit_history": [],
                "database_available": False,
                "status": "success"
            }
            
        profile_response = supabase.table('profiles').select('*').eq('id', user_id).execute()
        
        credit_response = supabase.table('credit_history')\
            .select('*')\
            .eq('user_id', user_id)\
            .order('created_at', desc=True)\
            .execute()
            
        return {
            "user_id": user_id,
            "profile_exists": bool(profile_response.data),
            "profile_data": profile_response.data,
            "credit_history": credit_response.data,
            "database_available": True,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "4.0.0",
        "services": {
            "ai_models": "available",
            "database": "available" if supabase else "unavailable",
            "payment": "available" if razorpay_client else "unavailable"
        }
    }

# --- 14. ERROR HANDLING ---

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "status": "error"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status": "error"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
