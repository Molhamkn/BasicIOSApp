import SwiftUI
import AVFoundation
import Vision
import PhotosUI
import CoreImage

struct CameraContainerView: View {
    @StateObject private var cameraManager = CameraManager()
    @State private var showCameraSwitcher = false
    @State private var baseZoom: CGFloat = 1.0
    @State private var detectedFaces: [FaceTarget] = []
    @State private var showTrainingMode = false
    
    var body: some View {
        ZStack {
            CameraPreviewViewRepresentable(cameraManager: cameraManager, detectedFaces: detectedFaces)
                .ignoresSafeArea()
                .gesture(
                    MagnificationGesture()
                        .onChanged { value in
                            let newZoom = baseZoom * value
                            cameraManager.setZoomImmediate(newZoom)
                        }
                        .onEnded { value in
                            baseZoom = cameraManager.zoom
                        }
                )
            
            IronManHUD(
                currentZoom: cameraManager.zoom,
                showCameraSwitcher: $showCameraSwitcher,
                cameraManager: cameraManager,
                faceCount: detectedFaces.count,
                showTrainingMode: $showTrainingMode
            )
            
            if showTrainingMode {
                TrainingModeView(cameraManager: cameraManager, showTrainingMode: $showTrainingMode)
            }
        }
        .onAppear {
            cameraManager.onFacesDetected = { faces in
                DispatchQueue.main.async {
                    detectedFaces = faces
                }
            }
            cameraManager.setup()
        }
    }
}

struct FaceTarget: Identifiable {
    let id = UUID()
    var rect: CGRect
    var confidence: Float = 1.0
    var recognizedName: String? = nil
}

class CameraManager: NSObject, ObservableObject {
    @Published var zoom: CGFloat = 1.0
    @Published var isFrontCamera: Bool = false
    @Published var isReady: Bool = false
    @Published var recognizedFaces: [String: String] = [:]
    
    let captureSession = AVCaptureSession()
    var currentInput: AVCaptureDeviceInput?
    var videoOutput: AVCaptureVideoDataOutput?
    var onFacesDetected: (([FaceTarget]) -> Void)?
    
    private var sequenceHandler: VNSequenceRequestHandler?
    private var frameCount = 0
    private let detectEveryNFrames = 3
    
    var faceClassifier: FaceClassifier?
    
    func setup() {
        sequenceHandler = VNSequenceRequestHandler()
        faceClassifier = FaceClassifier()
        faceClassifier?.load()
        setupCamera(position: .back)
    }
    
    func setupCamera(position: AVCaptureDevice.Position) {
        captureSession.beginConfiguration()
        captureSession.inputs.forEach { captureSession.removeInput($0) }
        
        if let videoOutput = videoOutput {
            captureSession.removeOutput(videoOutput)
        }
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            captureSession.commitConfiguration()
            return
        }
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
            currentInput = input
        }
        
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue", qos: .userInteractive))
        output.alwaysDiscardsLateVideoFrames = true
        
        if captureSession.canAddOutput(output) {
            captureSession.addOutput(output)
            videoOutput = output
        }
        
        captureSession.commitConfiguration()
        isFrontCamera = (position == .front)
        isReady = true
        
        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            if self?.captureSession.isRunning == false {
                self?.captureSession.startRunning()
            }
        }
    }
    
    func switchCamera() {
        let newPosition: AVCaptureDevice.Position = isFrontCamera ? .back : .front
        setupCamera(position: newPosition)
    }
    
    func setZoom(_ newZoom: CGFloat) {
        guard let input = currentInput else { return }
        let device = input.device
        
        let maxZoom = min(device.activeFormat.videoMaxZoomFactor, 10.0)
        let minZoom: CGFloat = 1.0
        zoom = max(minZoom, min(newZoom, maxZoom))
        
        do {
            try device.lockForConfiguration()
            device.videoZoomFactor = zoom
            device.unlockForConfiguration()
        } catch {}
    }
    
    func setZoomImmediate(_ newZoom: CGFloat) {
        guard let input = currentInput else { return }
        let device = input.device
        
        let maxZoom = min(device.activeFormat.videoMaxZoomFactor, 10.0)
        let minZoom: CGFloat = 1.0
        let clampedZoom = max(minZoom, min(newZoom, maxZoom))
        
        if abs(zoom - clampedZoom) > 0.01 {
            zoom = clampedZoom
            do {
                try device.lockForConfiguration()
                device.videoZoomFactor = clampedZoom
                device.unlockForConfiguration()
            } catch {}
        }
    }
    
    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        guard let handler = sequenceHandler else { return }
        
        frameCount += 1
        guard frameCount % detectEveryNFrames == 0 else { return }
        
        let detectRectangles = VNDetectFaceRectanglesRequest()
        let detectLandmarks = VNDetectFaceLandmarksRequest()
        
        try? handler.perform([detectRectangles, detectLandmarks], on: pixelBuffer)
        
        guard let rectangles = detectRectangles.results, !rectangles.isEmpty else {
            DispatchQueue.main.async { [weak self] in
                self?.onFacesDetected?([])
            }
            return
        }
        
        let faceTargets: [FaceTarget] = rectangles.compactMap { observation -> FaceTarget? in
            let name = faceClassifier?.recognize(observation: observation, pixelBuffer: pixelBuffer)
            return FaceTarget(
                rect: observation.boundingBox,
                confidence: observation.confidence,
                recognizedName: name
            )
        }
        
        DispatchQueue.main.async { [weak self] in
            self?.onFacesDetected?(faceTargets)
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processFrame(pixelBuffer)
    }
}

struct FaceEmbedding: Codable {
    let name: String
    let features: [Float]
    let createdAt: Date
}

class FaceClassifier {
    private var embeddings: [FaceEmbedding] = []
    private let minMatchConfidence: Float = 0.75
    private let maxStoredFacesPerPerson = 20
    
    private var storageURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("face_embeddings.json")
    }
    
    func load() {
        do {
            let data = try Data(contentsOf: storageURL)
            embeddings = try JSONDecoder().decode([FaceEmbedding].self, from: data)
        } catch {
            embeddings = []
        }
    }
    
    private func save() {
        do {
            let data = try JSONEncoder().encode(embeddings)
            try data.write(to: storageURL)
        } catch {
            print("Save error: \(error)")
        }
    }
    
    func train(cgImage: CGImage, personName: String) {
        let features = extractFeatures(from: cgImage)
        let embedding = FaceEmbedding(name: personName, features: features, createdAt: Date())
        embeddings.append(embedding)
        
        let count = embeddings.filter { $0.name == personName }.count
        if count > maxStoredFacesPerPerson {
            if let oldestIndex = embeddings.firstIndex(where: { $0.name == personName }) {
                embeddings.remove(at: oldestIndex)
            }
        }
        
        save()
    }
    
    private func extractFeatures(from cgImage: CGImage) -> [Float] {
        var features: [Float] = []
        
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
        
        guard let observation = request.results?.first else {
            return generateFallbackFeatures(from: cgImage)
        }
        
        if let landmarks = observation.landmarks {
            let bbox = observation.boundingBox
            features.append(Float(observation.confidence))
            
            let leftEyeCenter = getCenter(of: landmarks.leftEye?.normalizedPoints ?? [])
            let rightEyeCenter = getCenter(of: landmarks.rightEye?.normalizedPoints ?? [])
            let noseCenter = getCenter(of: landmarks.nose?.normalizedPoints ?? [])
            let mouthCenter = getCenter(of: landmarks.outerLips?.normalizedPoints ?? [])
            
            if let leftEye = landmarks.leftEye {
                features.append(contentsOf: normalizeLandmarkPoints(leftEye.normalizedPoints, bbox: bbox))
            } else {
                features.append(contentsOf: [Float](repeating: 0, count: 10))
            }
            
            if let rightEye = landmarks.rightEye {
                features.append(contentsOf: normalizeLandmarkPoints(rightEye.normalizedPoints, bbox: bbox))
            } else {
                features.append(contentsOf: [Float](repeating: 0, count: 10))
            }
            
            if let nose = landmarks.nose {
                features.append(contentsOf: normalizeLandmarkPoints(nose.normalizedPoints, bbox: bbox))
            } else {
                features.append(contentsOf: [Float](repeating: 0, count: 8))
            }
            
            if let outerLips = landmarks.outerLips {
                features.append(contentsOf: normalizeLandmarkPoints(outerLips.normalizedPoints, bbox: bbox))
            } else {
                features.append(contentsOf: [Float](repeating: 0, count: 18))
            }
            
            if let faceContour = landmarks.faceContour {
                features.append(contentsOf: normalizeLandmarkPoints(faceContour.normalizedPoints, bbox: bbox))
            } else {
                features.append(contentsOf: [Float](repeating: 0, count: 34))
            }
            
            features.append(Float(bbox.width))
            features.append(Float(bbox.height))
            features.append(Float(bbox.width / bbox.height))
            
            let eyeDistance = distance(leftEyeCenter, rightEyeCenter)
            let noseToMouth = distance(noseCenter, mouthCenter)
            let faceWidth = bbox.width
            let faceHeight = bbox.height
            
            features.append(Float(eyeDistance))
            features.append(Float(noseToMouth))
            features.append(Float(eyeDistance / faceWidth))
            features.append(Float(noseToMouth / faceHeight))
            features.append(Float(faceWidth / faceHeight))
            
            let leftEyeToNose = distance(leftEyeCenter, noseCenter)
            let rightEyeToNose = distance(rightEyeCenter, noseCenter)
            features.append(Float(leftEyeToNose / eyeDistance))
            features.append(Float(rightEyeToNose / eyeDistance))
            
            let leftEyebrow = landmarks.leftEyebrow?.normalizedPoints ?? []
            let rightEyebrow = landmarks.rightEyebrow?.normalizedPoints ?? []
            if !leftEyebrow.isEmpty {
                features.append(contentsOf: normalizeLandmarkPoints(leftEyebrow, bbox: bbox))
            } else {
                features.append(contentsOf: [Float](repeating: 0, count: 10))
            }
            if !rightEyebrow.isEmpty {
                features.append(contentsOf: normalizeLandmarkPoints(rightEyebrow, bbox: bbox))
            } else {
                features.append(contentsOf: [Float](repeating: 0, count: 10))
            }
        } else {
            return generateFallbackFeatures(from: cgImage)
        }
        
        while features.count < 128 {
            features.append(0)
        }
        
        return Array(features.prefix(128))
    }
    
    private func getCenter(of points: [CGPoint]) -> CGPoint {
        guard !points.isEmpty else { return .zero }
        let sumX = points.reduce(0) { $0 + $1.x }
        let sumY = points.reduce(0) { $0 + $1.y }
        return CGPoint(x: sumX / CGFloat(points.count), y: sumY / CGFloat(points.count))
    }
    
    private func distance(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        let dx = b.x - a.x
        let dy = b.y - a.y
        return sqrt(dx * dx + dy * dy)
    }
    
    private func normalizeLandmarkPoints(_ points: [CGPoint], bbox: CGRect) -> [Float] {
        return points.flatMap { pt -> [Float] in
            let x = Float((pt.x - bbox.origin.x) / bbox.width)
            let y = Float((pt.y - bbox.origin.y) / bbox.height)
            return [x, y]
        }
    }
    
    private func generateFallbackFeatures(from cgImage: CGImage) -> [Float] {
        let ciImage = CIImage(cgImage: cgImage)
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        
        let request = VNDetectFaceRectanglesRequest()
        try? handler.perform([request])
        
        guard let face = request.results?.first else {
            return [Float](repeating: 0, count: 128)
        }
        
        var features: [Float] = []
        features.append(Float(face.confidence))
        features.append(Float(face.boundingBox.origin.x))
        features.append(Float(face.boundingBox.origin.y))
        features.append(Float(face.boundingBox.width))
        features.append(Float(face.boundingBox.height))
        features.append(Float(face.boundingBox.width / face.boundingBox.height))
        
        while features.count < 128 {
            features.append(Float.random(in: 0...0.05))
        }
        
        return Array(features.prefix(128))
    }
    
    func recognize(observation: VNFaceObservation, pixelBuffer: CVPixelBuffer) -> String? {
        let features = extractFeaturesFromObservation(observation, pixelBuffer: pixelBuffer)
        guard !features.isEmpty else { return nil }
        
        var bestMatch: (name: String, similarity: Float) = ("", 0)
        
        for embedding in embeddings {
            let similarity = cosineSimilarity(features, embedding.features)
            if similarity > minMatchConfidence && similarity > bestMatch.similarity {
                bestMatch = (embedding.name, similarity)
            }
        }
        
        return bestMatch.similarity > minMatchConfidence ? bestMatch.name : nil
    }
    
    private func extractFeaturesFromObservation(_ observation: VNFaceObservation, pixelBuffer: CVPixelBuffer) -> [Float] {
        let bbox = observation.boundingBox
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        let cropX = Int(bbox.origin.x * CGFloat(width))
        let cropY = Int((1 - bbox.origin.y - bbox.height) * CGFloat(height))
        let cropWidth = Int(bbox.width * CGFloat(width))
        let cropHeight = Int(bbox.height * CGFloat(height))
        
        let cropRect = CGRect(x: max(0, cropX - cropWidth/4),
                              y: max(0, cropY - cropHeight/4),
                              width: min(cropWidth * 2, width - cropX + cropWidth/4),
                              height: min(cropHeight * 2, height - cropY + cropHeight/4))
        
        guard let cgImage = createCGImage(from: pixelBuffer, cropRect: cropRect) else {
            return generateFallbackFromBBox(observation)
        }
        
        return extractFeatures(from: cgImage)
    }
    
    private func generateFallbackFromBBox(_ observation: VNFaceObservation) -> [Float] {
        var features: [Float] = []
        features.append(Float(observation.confidence))
        features.append(Float(observation.boundingBox.origin.x))
        features.append(Float(observation.boundingBox.origin.y))
        features.append(Float(observation.boundingBox.width))
        features.append(Float(observation.boundingBox.height))
        features.append(Float(observation.boundingBox.width / observation.boundingBox.height))
        while features.count < 128 {
            features.append(Float.random(in: 0...0.05))
        }
        return Array(features.prefix(128))
    }
    
    private func createCGImage(from pixelBuffer: CVPixelBuffer, cropRect: CGRect) -> CGImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer).cropped(to: cropRect)
        let context = CIContext()
        return context.createCGImage(ciImage, from: ciImage.extent)
    }
    
    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        
        for i in 0..<min(a.count, b.count) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        
        let denominator = sqrt(normA) * sqrt(normB)
        guard denominator > 0 else { return 0 }
        
        return dotProduct / denominator
    }
    
    func getTrainedNames() -> [String] {
        var names: Set<String> = []
        for embedding in embeddings {
            names.insert(embedding.name)
        }
        return Array(names).sorted()
    }
    
    func clear() {
        embeddings = []
        save()
    }
}

struct TrainingModeView: View {
    @ObservedObject var cameraManager: CameraManager
    @Binding var showTrainingMode: Bool
    @State private var newPersonName = ""
    @State private var capturedImages: [UIImage] = []
    @State private var showImagePicker = false
    @State private var isTraining = false
    @State private var trainingComplete = false
    @State private var trainedPeople: [String] = []
    @State private var showClearAlert = false
    @State private var showCameraCapture = false
    
    var body: some View {
        VStack {
            HStack {
                Button(action: { showTrainingMode = false }) {
                    Image(systemName: "xmark")
                        .font(.system(size: 24))
                        .foregroundColor(.white)
                        .padding()
                }
                Spacer()
                Text("TRAINING MODE")
                    .font(.system(size: 18, weight: .bold, design: .monospaced))
                    .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                Spacer()
                Button(action: { showClearAlert = true }) {
                    Image(systemName: "trash")
                        .font(.system(size: 20))
                        .foregroundColor(.red)
                        .padding()
                }
            }
            .padding(.top, 60)
            
            Spacer()
            
            Text("Add photos of each person to recognize")
                .font(.system(size: 16))
                .foregroundColor(.white)
                .padding()
            
            TextField("Person Name", text: $newPersonName)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding(.horizontal, 40)
            
            HStack(spacing: 20) {
                Button(action: { showImagePicker = true }) {
                    VStack {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 30))
                        Text("Gallery")
                            .font(.caption)
                    }
                    .padding()
                    .background(Color(red: 0.0, green: 0.6, blue: 0.8))
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                
                Button(action: { showCameraCapture = true }) {
                    VStack {
                        Image(systemName: "camera")
                            .font(.system(size: 30))
                        Text("Capture")
                            .font(.caption)
                    }
                    .padding()
                    .background(Color(red: 0.0, green: 0.6, blue: 0.8))
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
            }
            .padding(.horizontal, 40)
            
            if !capturedImages.isEmpty {
                Text("\(capturedImages.count) photos selected")
                    .foregroundColor(.green)
                    .padding(.vertical, 5)
                
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(0..<capturedImages.count, id: \.self) { i in
                            Image(uiImage: capturedImages[i])
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: 60, height: 60)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                .overlay(
                                    Button(action: {
                                        capturedImages.remove(at: i)
                                    }) {
                                        Image(systemName: "xmark.circle.fill")
                                            .foregroundColor(.red)
                                    }
                                    .offset(x: 25, y: -25)
                                )
                        }
                    }
                    .padding(.horizontal)
                }
                .padding(.vertical, 5)
                
                Button(action: trainModel) {
                    if isTraining {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    } else {
                        Text("TRAIN: \(newPersonName)")
                    }
                }
                .padding()
                .background(Color.green)
                .foregroundColor(.white)
                .cornerRadius(10)
                .disabled(isTraining || newPersonName.isEmpty)
            }
            
            if trainingComplete {
                Text("Trained!")
                    .foregroundColor(.green)
                    .font(.headline)
                    .padding()
            }
            
            if !trainedPeople.isEmpty {
                VStack(alignment: .leading) {
                    Text("Trained People:")
                        .foregroundColor(.white)
                        .padding(.horizontal)
                    ForEach(trainedPeople, id: \.self) { person in
                        HStack {
                            Text("• \(person)")
                                .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                            Spacer()
                        }
                        .padding(.horizontal)
                    }
                }
                .padding(.top)
            }
            
            Spacer()
        }
        .background(Color.black.opacity(0.9))
        .sheet(isPresented: $showImagePicker) {
            ImagePicker(images: $capturedImages)
        }
        .sheet(isPresented: $showCameraCapture) {
            CameraCaptureView(capturedImages: $capturedImages)
        }
        .alert("Clear All Training?", isPresented: $showClearAlert) {
            Button("Cancel", role: .cancel) {}
            Button("Clear", role: .destructive) {
                cameraManager.faceClassifier?.clear()
                trainedPeople = []
            }
        }
        .onAppear {
            trainedPeople = cameraManager.faceClassifier?.getTrainedNames() ?? []
        }
    }
    
    func trainModel() {
        guard !newPersonName.isEmpty, !capturedImages.isEmpty else { return }
        
        isTraining = true
        
        DispatchQueue.global(qos: .userInitiated).async {
            for image in capturedImages {
                if let cgImage = image.cgImage {
                    cameraManager.faceClassifier?.train(cgImage: cgImage, personName: newPersonName)
                }
            }
            
            DispatchQueue.main.async {
                isTraining = false
                trainingComplete = true
                capturedImages = []
                newPersonName = ""
                trainedPeople = cameraManager.faceClassifier?.getTrainedNames() ?? []
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    trainingComplete = false
                }
            }
        }
    }
}

struct CameraCaptureView: UIViewControllerRepresentable {
    @Binding var capturedImages: [UIImage]
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        var parent: CameraCaptureView
        
        init(_ parent: CameraCaptureView) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let image = info[.originalImage] as? UIImage {
                DispatchQueue.main.async {
                    self.parent.capturedImages.append(image)
                }
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var images: [UIImage]
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.selectionLimit = 20
        config.filter = .images
        
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        var parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            parent.presentationMode.wrappedValue.dismiss()
            
            for result in results {
                result.itemProvider.loadObject(ofClass: UIImage.self) { object, error in
                    if let image = object as? UIImage {
                        DispatchQueue.main.async {
                            self.parent.images.append(image)
                        }
                    }
                }
            }
        }
    }
}

struct CameraPreviewViewRepresentable: UIViewRepresentable {
    @ObservedObject var cameraManager: CameraManager
    var detectedFaces: [FaceTarget]
    
    func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        view.backgroundColor = .black
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: cameraManager.captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = UIScreen.main.bounds
        view.layer.addSublayer(previewLayer)
        
        return view
    }
    
    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {
        DispatchQueue.main.async {
            uiView.layer.sublayers?.first?.frame = uiView.bounds
            
            if let previewLayer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer,
               let connection = previewLayer.connection {
                let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene
                let orientation = windowScene?.interfaceOrientation ?? .portrait
                switch orientation {
                case .portrait:
                    connection.videoOrientation = .portrait
                case .portraitUpsideDown:
                    connection.videoOrientation = .portraitUpsideDown
                case .landscapeLeft:
                    connection.videoOrientation = .landscapeLeft
                case .landscapeRight:
                    connection.videoOrientation = .landscapeRight
                default:
                    connection.videoOrientation = .portrait
                }
            }
            
            uiView.updateFaces(detectedFaces, bounds: uiView.bounds)
        }
    }
}

class CameraPreviewUIView: UIView {
    private var faceLayers: [CAShapeLayer] = []
    private var cornerLayers: [CAShapeLayer] = []
    private var nameLabels: [CATextLayer] = []
    
    func updateFaces(_ faces: [FaceTarget], bounds: CGRect) {
        faceLayers.forEach { $0.removeFromSuperlayer() }
        cornerLayers.forEach { $0.removeFromSuperlayer() }
        nameLabels.forEach { $0.removeFromSuperlayer() }
        faceLayers.removeAll()
        cornerLayers.removeAll()
        nameLabels.removeAll()
        
        for face in faces {
            let centerX = face.rect.midX * bounds.width
            let centerY = (1 - face.rect.midY) * bounds.height
            let size = max(face.rect.width, face.rect.height) * bounds.width
            
            let outerRadius = size / 2 + 8
            let innerRadius = size / 2 - 3
            
            let outerCircle = CAShapeLayer()
            outerCircle.path = UIBezierPath(arcCenter: CGPoint(x: centerX, y: centerY),
                                            radius: outerRadius,
                                            startAngle: 0,
                                            endAngle: .pi * 2,
                                            clockwise: true).cgPath
            outerCircle.strokeColor = UIColor(red: 0, green: 0.8, blue: 1, alpha: 1).cgColor
            outerCircle.fillColor = UIColor.clear.cgColor
            outerCircle.lineWidth = 1
            outerCircle.lineDashPattern = [4, 4]
            layer.addSublayer(outerCircle)
            faceLayers.append(outerCircle)
            
            let innerCircle = CAShapeLayer()
            innerCircle.path = UIBezierPath(arcCenter: CGPoint(x: centerX, y: centerY),
                                            radius: innerRadius,
                                            startAngle: 0,
                                            endAngle: .pi * 2,
                                            clockwise: true).cgPath
            innerCircle.strokeColor = UIColor(red: 0, green: 0.8, blue: 1, alpha: 1).cgColor
            innerCircle.fillColor = UIColor.clear.cgColor
            innerCircle.lineWidth = 2
            layer.addSublayer(innerCircle)
            faceLayers.append(innerCircle)
            
            let cornerSize: CGFloat = 12
            let corners: [(CGPoint, CGFloat)] = [
                (CGPoint(x: centerX - innerRadius, y: centerY - innerRadius), 0),
                (CGPoint(x: centerX + innerRadius, y: centerY - innerRadius), .pi / 2),
                (CGPoint(x: centerX + innerRadius, y: centerY + innerRadius), .pi),
                (CGPoint(x: centerX - innerRadius, y: centerY + innerRadius), .pi * 1.5)
            ]
            
            for (corner, startAngle) in corners {
                let cornerLayer = CAShapeLayer()
                let path = UIBezierPath()
                path.move(to: CGPoint(x: corner.x + cos(startAngle) * cornerSize,
                                      y: corner.y + sin(startAngle) * cornerSize))
                path.addLine(to: corner)
                path.addLine(to: CGPoint(x: corner.x + cos(startAngle + .pi / 2) * cornerSize,
                                         y: corner.y + sin(startAngle + .pi / 2) * cornerSize))
                cornerLayer.path = path.cgPath
                cornerLayer.strokeColor = UIColor(red: 0, green: 0.8, blue: 1, alpha: 1).cgColor
                cornerLayer.fillColor = UIColor.clear.cgColor
                cornerLayer.lineWidth = 2
                layer.addSublayer(cornerLayer)
                cornerLayers.append(cornerLayer)
            }
            
            let lineLength: CGFloat = 25
            let crosshairOffsets: [(CGFloat, CGFloat)] = [
                (0 as CGFloat, -innerRadius - lineLength),
                (0 as CGFloat, innerRadius + lineLength),
                (-innerRadius - lineLength, 0 as CGFloat),
                (innerRadius + lineLength, 0 as CGFloat)
            ]
            
            for (dx, dy) in crosshairOffsets {
                let lineLayer = CAShapeLayer()
                let path = UIBezierPath()
                path.move(to: CGPoint(x: centerX, y: centerY))
                path.addLine(to: CGPoint(x: centerX + dx, y: centerY + dy))
                lineLayer.path = path.cgPath
                lineLayer.strokeColor = UIColor(red: 0, green: 0.8, blue: 1, alpha: 0.8).cgColor
                lineLayer.lineWidth = 1
                layer.addSublayer(lineLayer)
                cornerLayers.append(lineLayer)
            }
            
            if let name = face.recognizedName {
                let nameLayer = CATextLayer()
                nameLayer.string = name
                nameLayer.fontSize = 14
                nameLayer.foregroundColor = UIColor(red: 1, green: 0.3, blue: 0, alpha: 1).cgColor
                nameLayer.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
                nameLayer.frame = CGRect(x: centerX - 50, y: centerY + innerRadius + 5, width: 100, height: 20)
                nameLayer.alignmentMode = .center
                layer.addSublayer(nameLayer)
                nameLabels.append(nameLayer)
            }
        }
    }
}

struct IronManHUD: View {
    let currentZoom: CGFloat
    @Binding var showCameraSwitcher: Bool
    let cameraManager: CameraManager
    var faceCount: Int = 0
    var showTrainingMode: Binding<Bool>? = nil
    
    var body: some View {
        VStack {
            HStack {
                TimeDisplay()
                    .padding(.leading, 20)
                    .padding(.top, 20)
                
                Spacer()
                
                ZoomIndicator(zoom: currentZoom)
                    .padding(.trailing, 20)
                    .padding(.top, 20)
                
                if let trainingBinding = showTrainingMode {
                    Button(action: { trainingBinding.wrappedValue.toggle() }) {
                        Image(systemName: "person.badge.plus")
                            .font(.system(size: 20))
                            .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                            .padding(.trailing, 10)
                    }
                }
            }
            
            Spacer()
            
            HStack {
                CameraSwitchButton(cameraManager: cameraManager, showCameraSwitcher: $showCameraSwitcher)
                    .padding(.leading, 20)
                    .padding(.bottom, 20)
                
                Spacer()
                
                if faceCount > 0 {
                    TargetCounter(count: faceCount)
                        .padding(.trailing, 20)
                        .padding(.bottom, 20)
                }
            }
        }
    }
}

struct TargetCounter: View {
    let count: Int
    @State private var pulse = false
    
    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: "scope")
                .font(.system(size: 14))
            Text("TARGETS: \(count)")
                .font(.system(size: 14, weight: .bold, design: .monospaced))
        }
        .foregroundColor(.red)
        .shadow(color: .red, radius: pulse ? 10 : 5)
        .scaleEffect(pulse ? 1.05 : 1.0)
        .onAppear {
            withAnimation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true)) {
                pulse = true
            }
        }
    }
}

struct TimeDisplay: View {
    @State private var currentTime = Date()
    let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    
    var body: some View {
        Text(formattedTime)
            .font(.system(size: 24, weight: .light, design: .monospaced))
            .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
            .shadow(color: Color(red: 0.0, green: 0.5, blue: 1.0), radius: 5)
            .onReceive(timer) { _ in
                currentTime = Date()
            }
    }
    
    private var formattedTime: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm:ss a"
        return formatter.string(from: currentTime)
    }
}

struct ZoomIndicator: View {
    let zoom: CGFloat
    
    var body: some View {
        Text(String(format: "%.1fx", zoom))
            .font(.system(size: 18, weight: .light, design: .monospaced))
            .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
            .shadow(color: Color(red: 0.0, green: 0.5, blue: 1.0), radius: 3)
    }
}

struct CameraSwitchButton: View {
    let cameraManager: CameraManager
    @Binding var showCameraSwitcher: Bool
    
    var body: some View {
        Button(action: {
            cameraManager.switchCamera()
            showCameraSwitcher = false
        }) {
            Image(systemName: "camera.rotate")
                .font(.system(size: 24))
                .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                .shadow(color: Color(red: 0.0, green: 0.5, blue: 1.0), radius: 5)
                .frame(width: 50, height: 50)
        }
    }
}

@main
struct BasicIOSAppApp: App {
    var body: some Scene {
        WindowGroup {
            CameraContainerView()
        }
    }
}
