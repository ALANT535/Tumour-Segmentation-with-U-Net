from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the saved CNN model
model = None
try:
    model = tf.keras.models.load_model('/home/rohitb/projects/del/del1/SAVED CNN MODEL FILES/saved_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")

@csrf_exempt
def predict_image(request):
    if request.method == 'POST':
        try:
            if model is None:
                return JsonResponse({
                    'error': 'Model not loaded', 
                    'status': 'error'
                }, status=500)

            # Get the uploaded image from the request
            image_file = request.FILES['image']
            
            # Read and preprocess the image
            image = Image.open(image_file)
            image = image.resize((150, 150))  # Changed to 150x150 to match expected input
            image = np.array(image)
            
            # Ensure image has 3 channels (RGB)
            if len(image.shape) == 2:  # If grayscale, convert to RGB
                image = np.stack((image,)*3, axis=-1)
            elif image.shape[-1] == 4:  # If RGBA, convert to RGB
                image = image[:,:,:3]
                
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            
            # Make prediction
            prediction = model.predict(image)
            
            # Process prediction result
            result = {
                'prediction': prediction.tolist(),
                'status': 'success'
            }
            
            return JsonResponse(result)
            
        except Exception as e:
            return JsonResponse({
                'error': str(e), 
                'status': 'error'
            }, status=400)
            
    return JsonResponse({
        'error': 'Only POST requests are allowed', 
        'status': 'error'
    }, status=405)