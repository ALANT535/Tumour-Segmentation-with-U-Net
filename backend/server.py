from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the saved CNN model
model = tf.keras.models.load_model('./SAVED CNN MODEL FILES')

@csrf_exempt
def predict_image(request):
    if request.method == 'POST':
        try:
            # Get the uploaded image from the request
            image_file = request.FILES['image']
            
            # Read and preprocess the image
            image = Image.open(image_file)
            image = image.resize((224, 224))  # Adjust size according to your model's input requirements
            image = np.array(image)
            image = image / 255.0  # Normalize pixel values
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            
            # Make prediction
            prediction = model.predict(image)
            
            # Process prediction result
            # Modify this part according to your model's output format
            result = {
                'prediction': prediction.tolist(),
                'status': 'success'
            }
            
            return JsonResponse(result)
            
        except Exception as e:
            return JsonResponse({'error': str(e), 'status': 'error'}, status=400)
            
    return JsonResponse({'error': 'Only POST requests are allowed', 'status': 'error'}, status=405)
