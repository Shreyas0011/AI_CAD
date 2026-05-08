# Vercel Deployment Instructions

To successfully deploy this project to Vercel, please follow these settings in the Vercel Dashboard:

1.  **Framework Preset**: Other (or Vite if it detects it)
2.  **Root Directory**: `.` (The repository root)
3.  **Build Command**: `cd frontend && npm install && npm run build`
4.  **Output Directory**: `frontend/dist`
5.  **Install Command**: `npm install`

### Backend Notes
The backend is powered by FastAPI and is served from the `api/` directory. 
We are using `tensorflow-cpu` to stay within Vercel's serverless function size limits.

If you encounter "Function Size Limit Exceeded", the TensorFlow model may be too large for Vercel Serverless. In that case, consider hosting the backend on Render.com or Railway.app.
