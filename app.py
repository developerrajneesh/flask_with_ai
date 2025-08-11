# # # # # # # from flask import Flask, jsonify

# # # # # # # app = Flask(__name__)

# # # # # # # # Route 1 - Simple text response
# # # # # # # @app.route("/")
# # # # # # # def home():
# # # # # # #     return "Welcome to my Flask App!"

# # # # # # # # Route 2 - JSON API response
# # # # # # # @app.route("/api/data")
# # # # # # # def get_data():
# # # # # # #     data = {
# # # # # # #         "name": "John Doe",
# # # # # # #         "age": 25,
# # # # # # #         "city": "New York"
# # # # # # #     }
# # # # # # #     return jsonify(data)

# # # # # # # if __name__ == "__main__":
# # # # # # #     app.run(debug=True)

# # # # # # from flask import Flask, request, jsonify
# # # # # # from transformers import pipeline

# # # # # # app = Flask(__name__)

# # # # # # # Load free pre-trained model locally
# # # # # # model = pipeline("sentiment-analysis")  

# # # # # # @app.route("/analyze", methods=["POST"])
# # # # # # def analyze():
# # # # # #     text = request.json.get("text", "")
# # # # # #     result = model(text)    
# # # # # #     return jsonify(result)

# # # # # # if __name__ == "__main__":
# # # # # #     app.run(debug=True)
# # # # # import io
# # # # # import requests
# # # # # from flask import Flask, request, jsonify
# # # # # from PIL import Image
# # # # # import torch
# # # # # import torchvision.transforms as transforms
# # # # # import torchvision.models as models

# # # # # app = Flask(__name__)

# # # # # # Load pre-trained ResNet50 model
# # # # # model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# # # # # model.eval()

# # # # # # Load ImageNet labels (download once and store locally)
# # # # # LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# # # # # imagenet_classes = requests.get(LABELS_URL).text.splitlines()

# # # # # # Image transformation
# # # # # transform = transforms.Compose([
# # # # #     transforms.Resize(256),
# # # # #     transforms.CenterCrop(224),
# # # # #     transforms.ToTensor(),
# # # # #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
# # # # #                          std=[0.229, 0.224, 0.225])
# # # # # ])

# # # # # @app.route("/classify-image", methods=["POST"])
# # # # # def classify_image():
# # # # #     if 'image' not in request.files:
# # # # #         return jsonify({"error": "No image uploaded"}), 400
    
# # # # #     image_file = request.files['image']
# # # # #     image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    
# # # # #     img_t = transform(image).unsqueeze(0)

# # # # #     with torch.no_grad():
# # # # #         outputs = model(img_t)
# # # # #         _, predicted = outputs.max(1)
# # # # #         class_name = imagenet_classes[predicted.item()]
    
# # # # #     return jsonify({"predicted_class": class_name})

# # # # # if __name__ == "__main__":
# # # # #     app.run(debug=True)
# # # # import io
# # # # import requests
# # # # from flask import Flask, request, jsonify
# # # # from PIL import Image
# # # # import torch
# # # # import torchvision.transforms as transforms
# # # # import torchvision.models as models

# # # # app = Flask(__name__)

# # # # # Load MobileNetV2 pre-trained on ImageNet
# # # # model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# # # # model.eval()

# # # # # Load ImageNet class labels
# # # # LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# # # # imagenet_classes = requests.get(LABELS_URL).text.splitlines()

# # # # # Image transformation pipeline
# # # # transform = transforms.Compose([
# # # #     transforms.Resize(256),
# # # #     transforms.CenterCrop(224),
# # # #     transforms.ToTensor(),
# # # #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
# # # #                          std=[0.229, 0.224, 0.225])
# # # # ])

# # # # @app.route("/classify-image", methods=["POST"])
# # # # def classify_image():
# # # #     if 'image' not in request.files:
# # # #         return jsonify({"error": "No image uploaded"}), 400
    
# # # #     image_file = request.files['image']
# # # #     image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    
# # # #     img_t = transform(image).unsqueeze(0)  # Add batch dimension

# # # #     with torch.no_grad():
# # # #         outputs = model(img_t)
# # # #         probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
# # # #         top5_prob, top5_catid = torch.topk(probabilities, 5)

# # # #     results = []
# # # #     for i in range(top5_prob.size(0)):
# # # #         results.append({
# # # #             "label": imagenet_classes[top5_catid[i]],
# # # #             "confidence": round(top5_prob[i].item() * 100, 2)
# # # #         })
    
# # # #     return jsonify({"predictions": results})

# # # # if __name__ == "__main__":
# # # #     app.run(debug=True)

# # # from flask import Flask, request, jsonify
# # # from transformers import pipeline
# # # import torch
# # # import torchvision.models as models
# # # import torchvision.transforms as transforms
# # # from PIL import Image
# # # import requests
# # # from io import BytesIO

# # # app = Flask(__name__)

# # # # Load MobileNetV2 for image classification
# # # imagenet_classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# # # imagenet_classes = requests.get(imagenet_classes_url).text.splitlines()

# # # mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# # # mobilenet.eval()

# # # transform = transforms.Compose([
# # #     transforms.Resize(256),
# # #     transforms.CenterCrop(224),
# # #     transforms.ToTensor(),
# # #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
# # #                          std=[0.229, 0.224, 0.225])
# # # ])

# # # # Load GPT-2 for text generation
# # # gpt2_generator = pipeline("text-generation", model="gpt2")

# # # @app.route("/classify-and-describe", methods=["POST"])
# # # def classify_and_describe():
# # #     try:
# # #         data = request.get_json()
# # #         image_url = data.get("image_url")
# # #         if not image_url:
# # #             return jsonify({"error": "image_url is required"}), 400

# # #         # Load and preprocess image
# # #         response = requests.get(image_url)
# # #         image = Image.open(BytesIO(response.content)).convert("RGB")
# # #         img_t = transform(image).unsqueeze(0)

# # #         # Predict class
# # #         with torch.no_grad():
# # #             outputs = mobilenet(img_t)
# # #             _, predicted_idx = outputs.max(1)
# # #         label = imagenet_classes[predicted_idx]

# # #         # Use GPT-2 to describe the image
# # #         prompt = f"This is a photo of a {label}. It looks"
# # #         gpt_output = gpt2_generator(prompt, max_length=40, num_return_sequences=1)
# # #         description = gpt_output[0]['generated_text']

# # #         return jsonify({
# # #             "predicted_label": label,
# # #             "gpt2_description": description
# # #         })

# # #     except Exception as e:
# # #         return jsonify({"error": str(e)}), 500

# # # if __name__ == "__main__":
# # #     app.run(debug=True)


from flask import Flask, request, jsonify
from transformers import pipeline
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load MobileNetV2 for image classification
imagenet_classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
import requests
imagenet_classes = requests.get(imagenet_classes_url).text.splitlines()

mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
mobilenet.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load GPT-2 for text generation
gpt2_generator = pipeline("text-generation", model="gpt2")

@app.route("/classify-and-describe", methods=["POST"])
def classify_and_describe():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Image file is required"}), 400

        # Get optional custom question prompt
        custom_prompt = request.form.get("prompt", "")

        # Load uploaded image
        image_file = request.files["image"]
        image = Image.open(image_file.stream).convert("RGB")
        img_t = transform(image).unsqueeze(0)

        # Predict class with MobileNetV2
        with torch.no_grad():
            outputs = mobilenet(img_t)
            _, predicted_idx = outputs.max(1)
        label = imagenet_classes[predicted_idx]

        # Build GPT-2 prompt
        if custom_prompt:
            prompt = f"The image shows a {label}. {custom_prompt}"
        else:
            prompt = f"This is a photo of a {label}. It looks"

        # Generate text
        gpt_output = gpt2_generator(prompt, max_length=60, num_return_sequences=1)
        description = gpt_output[0]['generated_text']

        return jsonify({
            "predicted_label": label,
            "gpt2_response": description
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

