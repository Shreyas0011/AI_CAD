import sys
import os

# Add the project root to the path so we can import 'backend'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app

# Export the app for Vercel
# Vercel's @vercel/python runtime will look for 'app' or 'handler'
# In FastAPI, it's 'app'
