import os
from groq import Groq
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Set your API key for the language model
api_key = "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"

# Initialize FastAPI app
app = FastAPI()

# # Configuration class
# class CFG:
#     device = "cpu"  # Use CPU for inference
#     seed = 42
#     generator = torch.Generator(device).manual_seed(seed)  # Ensures reproducibility
#     image_gen_steps = 35  # Number of diffusion steps for image generation
#     image_gen_model_id = "stabilityai/stable-diffusion-2"  # Model ID from Hugging Face
#     image_gen_size = (400, 400)  # Desired image size
#     image_gen_guidance_scale = 9  # Strength of the guidance signal


# # Access Hugging Face API token
# secret_hf_token = "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"  # Replace with actual token

# # Load the pre-trained Stable Diffusion model once on app startup
# @app.on_event("startup")
# async def load_image_model():
#     global image_gen_model
#     image_gen_model = StableDiffusionPipeline.from_pretrained(
#         CFG.image_gen_model_id,
#         torch_dtype=torch.float32,  # Use full precision for CPU
#         use_auth_token=secret_hf_token,
#     ).to(CFG.device)


# # Define request model for image generation
# class ImageGenerationRequest(BaseModel):
#     prompt: str


# # Image generation function
# from fastapi.responses import StreamingResponse
# from io import BytesIO

# # Image generation function
# @app.post("/generate_image/")
# async def generate_image(request: ImageGenerationRequest) -> StreamingResponse:
#     prompt = request.prompt
#     # Generate image based on the prompt
#     image = image_gen_model(
#         prompt,
#         num_inference_steps=CFG.image_gen_steps,
#         generator=CFG.generator,
#         guidance_scale=CFG.image_gen_guidance_scale,
#     ).images[0]

#     # Resize the generated image to the desired size
#     image = image.resize(CFG.image_gen_size)

#     # Save image to a BytesIO stream
#     img_byte_arr = BytesIO()
#     image.save(img_byte_arr, format='PNG')
#     img_byte_arr.seek(0)

#     # Return the image as a streaming response
#     return StreamingResponse(img_byte_arr, media_type="image/png")



# Set your API key
api_key = "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"

# Initialize the Groq client
client = Groq(api_key=api_key)

# app = FastAPI()

# Define model details (replace with actual model in use)
model = "llama-3.1-70b-versatile"  # Replace with the actual model identifier for your LLM

# Request models
class ProductDescriptionRequest(BaseModel):
    product_description: str

class BusinessModelRequest(BaseModel):
    product_description: str
    sustainability_report: str

class MarketFitRequest(BaseModel):
    product_description: str
    business_model: str

class MarketingStrategyRequest(BaseModel):
    product_description: str
    market_fit: str

class BusinessPlanRequest(BaseModel):
    sustainability_report: str
    business_model: str
    market_fit: str
    marketing_strategy: str

class ESGMetricsRequest(BaseModel):
    product_description: str

# 1. Sustainability API
@app.post("/sustainability/")
async def sustainability_llm(request: ProductDescriptionRequest):
    messages = [
        {
            "role": "user",
            "content": f"Given the product: {request.product_description}, suggest ways to ensure sustainability (Recycle, Reuse, Reduce)."
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.7,
        max_tokens=500,
    )
    
    response_text = chat_completion.choices[0].message.content
    return {"sustainability_suggestions": response_text.strip()}

# 2. Business Model Advisor API (with Circular Economy Layer)
@app.post("/business_model/")
async def business_model_llm(request: BusinessModelRequest):
    messages = [
        {
            "role": "user",
            "content": (
                f"Given the product: {request.product_description}, and sustainability strategy: {request.sustainability_report}, "
                "recommend a circular economy business model (e.g., leasing, subscription services, repair/reuse schemes) and analyze profitability and sustainability impacts."
            )
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.7,
        max_tokens=500,
    )
    
    response_text = chat_completion.choices[0].message.content
    return {"business_model_suggestions": response_text.strip()}

# 3. Market Research & Product-Market Fit API
@app.post("/market_fit/")
async def market_fit_llm(request: MarketFitRequest):
    messages = [
        {
            "role": "user",
            "content": f"Given the product: {request.product_description}, and the business model: {request.business_model}, suggest the target market and analyze the product-market fit."
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.7,
        max_tokens=500,
    )

    response_text = chat_completion.choices[0].message.content
    return {"market_fit_analysis": response_text.strip()}

# 4. Marketing Strategy API
@app.post("/marketing_strategy/")
async def marketing_strategy_llm(request: MarketingStrategyRequest):
    messages = [
        {
            "role": "user",
            "content": f"Given the product: {request.product_description}, and the market fit analysis: {request.market_fit}, propose innovative marketing strategies."
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.7,
        max_tokens=500,
    )

    response_text = chat_completion.choices[0].message.content
    return {"marketing_strategy": response_text.strip()}

# 5. Business Plan Documentation API
@app.post("/business_plan/")
async def business_plan_llm(request: BusinessPlanRequest):
    messages = [
        {
            "role": "user",
            "content": (
                f"Create a comprehensive business plan based on the following:\n"
                f"Sustainability Report: {request.sustainability_report}\n"
                f"Business Model: {request.business_model}\n"
                f"Market Fit: {request.market_fit}\n"
                f"Marketing Strategy: {request.marketing_strategy}"
            )
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.7,
        max_tokens=1000,
    )

    response_text = chat_completion.choices[0].message.content
    return {"business_plan": response_text.strip()}

# 6. Advanced Sustainability Insights using AI-driven ESG Metrics
@app.post("/esg_metrics/")
async def esg_metrics_llm(request: ESGMetricsRequest):
    messages = [
        {
            "role": "user",
            "content": (
                f"Generate detailed ESG metrics for the product lifecycle of {request.product_description}, "
                "including carbon footprint, water usage, energy efficiency, and supply chain risks."
            )
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.7,
        max_tokens=700,
    )

    response_text = chat_completion.choices[0].message.content
    return {"esg_metrics": response_text.strip()}

# 7. Impact Prediction Models for Long-Term Strategy Assessment
@app.post("/impact_prediction/")
async def impact_prediction_llm(request: ESGMetricsRequest):
    messages = [
        {
            "role": "user",
            "content": (
                f"Predict the long-term environmental and social impacts of sustainability strategies for {request.product_description}. "
                "Assess the effects on future regulations, market trends, and social values."
            )
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.7,
        max_tokens=700,
    )

    response_text = chat_completion.choices[0].message.content
    return {"impact_prediction": response_text.strip()}

# 8. Orchestrate the Complete Business Plan Generation
@app.post("/generate_business_plan/")
async def generate_business_plan(product_description: str):
    try:
        BASE_URL = "http://localhost:8000"  # Update with the actual base URL if needed
        
        # Call Sustainability API
        sustainability_response = requests.post(f"{BASE_URL}/sustainability/", json={"product_description": product_description})
        sustainability_report = sustainability_response.json()['sustainability_suggestions']
        
        # Call Business Model API
        business_model_response = requests.post(f"{BASE_URL}/business_model/", json={"product_description": product_description, "sustainability_report": sustainability_report})
        business_model = business_model_response.json()['business_model_suggestions']
        
        # Call Market Fit API
        market_fit_response = requests.post(f"{BASE_URL}/market_fit/", json={"product_description": product_description, "business_model": business_model})
        market_fit = market_fit_response.json()['market_fit_analysis']
        
        # Call Marketing Strategy API
        marketing_strategy_response = requests.post(f"{BASE_URL}/marketing_strategy/", json={"product_description": product_description, "market_fit": market_fit})
        marketing_strategy = marketing_strategy_response.json()['marketing_strategy']
        
        # Call Business Plan API
        business_plan_response = requests.post(f"{BASE_URL}/business_plan/", json={
            "sustainability_report": sustainability_report,
            "business_model": business_model,
            "market_fit": market_fit,
            "marketing_strategy": marketing_strategy
        })
        business_plan = business_plan_response.json()['business_plan']
        
        return {"business_plan": business_plan}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
