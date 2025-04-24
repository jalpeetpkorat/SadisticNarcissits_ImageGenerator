from flask import Flask, request, jsonify, render_template
from diffusers import StableDiffusionPipeline
import torch
import os

# Initialize Flask app
app = Flask(__name__)

# Load Stable Diffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.json
    prompt = data.get("prompt", "A fantasy landscape with dragons")
    
    # Generate image
    image = model(prompt).images[0]
    image_path = "static/generated_image.png"
    image.save(image_path)
    
    return jsonify({"image_url": image_path})

if __name__ == "__main__":
    app.run(debug=True)