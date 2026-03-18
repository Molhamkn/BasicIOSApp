# Iron Man HUD - ML Person Recognition

## Current Status

The app currently uses Apple's Vision framework for face detection. Vision already uses Core ML under the hood, but we can enhance it with a custom model for person recognition.

## How to Train a Custom ML Model

### Option 1: Create ML (Recommended)

1. Open Xcode
2. Go to **Xcode > Developer Tools > Create ML**
3. Create a new project with **Image Classification** template
4. Add training images for each person you want to recognize:
   ```
   Training/
   ├── person1_name/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── person2_name/
   │   ├── image1.jpg
   │   └── ...
   └── unknown/
       └── (images of random people)
   ```
5. Train the model
6. Export as `.mlmodel` file
7. Add to the app in Xcode

### Option 2: Turi Create (More Control)

```bash
pip install turicreate
```

```python
import turicreate as tc

# Load training images
data = tc.image_analysis.load_images('training_data/', with_path=True)

# Add labels from folder names
data['label'] = data['path'].apply(lambda path: path.split('/')[-2])

# Split data
train_data, test_data = data.random_split(0.8)

# Create and train model
model = tc.image_classifier.create(train_data, target='label', model='squeezenet_v1.1')

# Save model
model.save('PersonClassifier.mlmodel')
```

### Option 3: Use Pre-trained Model

Download a pre-trained model from Apple's model gallery:
- Visit https://developer.apple.com/machine-learning/models/
- Download an appropriate model
- Convert if needed using coremltools

## Integrating Your Model

Once you have a `.mlmodel` file:

1. Add it to your Xcode project
2. Update `CameraManager.swift`:

```swift
private var mlModel: VNCoreMLModel?

private func setupMLModel() {
    do {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
        // Replace with your model
        let model = try YourModel(configuration: config)
        mlModel = try VNCoreMLModel(for: model.model)
        
    } catch {
        print("ML Model setup error: \(error)")
    }
}

func recognizePerson(faceImage: CGImage, completion: @escaping (String?) -> Void) {
    guard let model = mlModel else {
        completion(nil)
        return
    }
    
    let request = VNCoreMLRequest(model: model) { request, error in
        guard let results = request.results as? [VNClassificationObservation] else {
            completion(nil)
            return
        }
        
        if let topResult = results.first, topResult.confidence > 0.7 {
            completion(topResult.identifier)
        } else {
            completion(nil)
        }
    }
    
    let handler = VNImageRequestHandler(cgImage: faceImage)
    try? handler.perform([request])
}
```

## Tips for Best Results

1. **Use 50+ images per person** for good accuracy
2. **Variety matters**: different angles, lighting, expressions
3. **Include "unknown" category** for people you don't want to recognize
4. **Test with real-world conditions** before finalizing

## Model Performance

- Smaller models = faster but less accurate
- Larger models = more accurate but slower
- For real-time HUD, use `squeezenet` or `mobilenet` architectures
