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
    @State private var showSettings = false
    @State private var showChat = false
    
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
                showTrainingMode: $showTrainingMode,
                showSettings: $showSettings,
                showChat: $showChat
            )
            
            if showTrainingMode {
                TrainingModeView(cameraManager: cameraManager, showTrainingMode: $showTrainingMode)
            }
            
            if showSettings {
                SettingsView(showSettings: $showSettings)
            }
            
            if showChat {
                ChatView(showChat: $showChat)
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
    let id: Int
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
    private var trackedFaces: [(id: Int, observation: VNFaceObservation, name: String?)] = []
    private var nextFaceId = 0
    private var frameCount = 0
    private let detectEveryNFrames = 1
    
    private var openRouterClient = OpenRouterClient()
    private var lastApiCall: Date = .distantPast
    private let apiCooldown: TimeInterval = 5.0
    private var lastRecognizedNames: Set<String> = []
    
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
        trackedFaces = []
        nextFaceId = 0
        
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
        
        var faceTargets: [FaceTarget] = []
        
        if frameCount % detectEveryNFrames == 0 || trackedFaces.isEmpty {
            let detectRequest = VNDetectFaceRectanglesRequest()
            let landmarksRequest = VNDetectFaceLandmarksRequest()
            try? handler.perform([detectRequest, landmarksRequest], on: pixelBuffer)
            
            if let rectangles = detectRequest.results, !rectangles.isEmpty {
                let detectedRects = rectangles.map { $0.boundingBox }
                let matchedFaces = matchFaces(oldFaces: trackedFaces, newRects: detectedRects)
                
                trackedFaces = rectangles.enumerated().map { index, rect in
                    let name = faceClassifier?.recognize(observation: rect, pixelBuffer: pixelBuffer)
                    let id = matchedFaces.first { $0.newIndex == index }?.id ?? nextFaceId
                    if !matchedFaces.contains(where: { $0.id == id && $0.newIndex == index }) && !trackedFaces.contains(where: { $0.id == id }) {
                        nextFaceId += 1
                    }
                    return (id: id, observation: rect, name: name)
                }
                
                nextFaceId = (trackedFaces.map { $0.id }.max() ?? 0) + 1
                
                // Call OpenRouter for better recognition
                callOpenRouterForRecognition(pixelBuffer: pixelBuffer)
            } else {
                trackedFaces = []
            }
        }
        
        faceTargets = trackedFaces.map { tracked in
            FaceTarget(id: tracked.id, rect: tracked.observation.boundingBox, confidence: tracked.observation.confidence, recognizedName: tracked.name)
        }
        
        DispatchQueue.main.async { [weak self] in
            self?.onFacesDetected?(faceTargets)
        }
    }
    
    private func callOpenRouterForRecognition(pixelBuffer: CVPixelBuffer) {
        guard Date().timeIntervalSince(lastApiCall) > apiCooldown else { return }
        guard !trackedFaces.isEmpty else { return }
        
        lastApiCall = Date()
        
        let knownNames = faceClassifier?.getTrainedNames() ?? []
        
        guard let cgImage = createCGImage(from: pixelBuffer) else { return }
        let image = UIImage(cgImage: cgImage)
        
        openRouterClient.identifyFace(image: image, knownNames: knownNames) { [weak self] identifiedName in
            guard let name = identifiedName else { return }
            
            // Update the tracked face with the name from OpenRouter
            if let index = self?.trackedFaces.firstIndex(where: { $0.name == nil || $0.name?.isEmpty == true }) {
                self?.trackedFaces[index].name = name
                Jarvis.shared.announceFaceRecognized(name: name)
            }
        }
    }
    
    private func createCGImage(from pixelBuffer: CVPixelBuffer) -> CGImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        return context.createCGImage(ciImage, from: ciImage.extent)
    }
    
    private func matchFaces(oldFaces: [(id: Int, observation: VNFaceObservation, name: String?)], newRects: [CGRect]) -> [(id: Int, newIndex: Int)] {
        var matches: [(id: Int, newIndex: Int)] = []
        
        for (newIndex, newRect) in newRects.enumerated() {
            var bestMatch: (oldIndex: Int, iou: CGFloat) = (-1, 0)
            
            for (oldIndex, old) in oldFaces.enumerated() {
                let iou = calculateIoU(old.observation.boundingBox, newRect)
                if iou > bestMatch.iou && iou > 0.3 {
                    bestMatch = (oldIndex, iou)
                }
            }
            
            if bestMatch.oldIndex >= 0 {
                matches.append((id: oldFaces[bestMatch.oldIndex].id, newIndex: newIndex))
            }
        }
        
        return matches
    }
    
    private func calculateIoU(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        let interArea = intersection.width * intersection.height
        let aArea = a.width * a.height
        let bArea = b.width * b.height
        let unionArea = aArea + bArea - interArea
        guard unionArea > 0 else { return 0 }
        return interArea / unionArea
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
    private let minMatchConfidence: Float = 0.83
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
    
    func recognize(cgImage: CGImage) -> String? {
        let features = extractFeatures(from: cgImage)
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
    
    private var displayRects: [Int: (rect: CGRect, name: String, alpha: CGFloat)] = [:]
    private let fadeSpeed: CGFloat = 0.15
    
    private var displayLink: CADisplayLink?
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        startAnimation()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        startAnimation()
    }
    
    private func startAnimation() {
        displayLink = CADisplayLink(target: self, selector: #selector(updateAnimation))
        displayLink?.preferredFrameRateRange = CAFrameRateRange(minimum: 60, maximum: 120)
        displayLink?.add(to: .main, forMode: .common)
    }
    
    @objc private func updateAnimation() {
        var needsUpdate = false
        
        // Fade out faces that are gone
        let idsToRemove = displayRects.keys.filter { id in
            targetRects[id] == nil
        }
        for id in idsToRemove {
            if let current = displayRects[id] {
                let newAlpha = current.alpha - fadeSpeed
                if newAlpha <= 0 {
                    displayRects.removeValue(forKey: id)
                } else {
                    displayRects[id] = (rect: current.rect, name: current.name, alpha: newAlpha)
                }
                needsUpdate = true
            }
        }
        
        // Update positions and fade in new faces
        for (id, target) in targetRects {
            if let current = displayRects[id] {
                // Update position instantly, fade in if needed
                let newAlpha = current.alpha < 1 ? min(1, current.alpha + fadeSpeed) : current.alpha
                displayRects[id] = (rect: target.rect, name: target.name, alpha: newAlpha)
            } else {
                // New face - start with fade in
                displayRects[id] = (rect: target.rect, name: target.name, alpha: 0)
            }
            needsUpdate = true
        }
        
        if needsUpdate {
            updateLayers()
        }
    }
    
    private var targetRects: [Int: (rect: CGRect, name: String)] = [:]
    
    func updateFaces(_ faces: [FaceTarget], bounds: CGRect) {
        targetRects = Dictionary(uniqueKeysWithValues: faces.map { ($0.id, ($0.rect, $0.recognizedName ?? "")) })
    }
    
    private func updateLayers() {
        faceLayers.forEach { $0.removeFromSuperlayer() }
        cornerLayers.forEach { $0.removeFromSuperlayer() }
        nameLabels.forEach { $0.removeFromSuperlayer() }
        faceLayers.removeAll()
        cornerLayers.removeAll()
        nameLabels.removeAll()
        
        let bounds = self.bounds
        
        for (_, display) in displayRects {
            guard display.alpha > 0 else { continue }
            
            let centerX = display.rect.midX * bounds.width
            let centerY = (1 - display.rect.midY) * bounds.height
            let size = max(display.rect.width, display.rect.height) * bounds.width
            
            let outerRadius = size / 2 + 8
            let innerRadius = size / 2 - 3
            
            let outerCircle = CAShapeLayer()
            outerCircle.path = UIBezierPath(arcCenter: CGPoint(x: centerX, y: centerY),
                                            radius: outerRadius,
                                            startAngle: 0,
                                            endAngle: .pi * 2,
                                            clockwise: true).cgPath
            outerCircle.strokeColor = UIColor(red: 0, green: 0.8, blue: 1, alpha: display.alpha).cgColor
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
            innerCircle.strokeColor = UIColor(red: 0, green: 0.8, blue: 1, alpha: display.alpha).cgColor
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
                cornerLayer.strokeColor = UIColor(red: 0, green: 0.8, blue: 1, alpha: display.alpha).cgColor
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
                lineLayer.strokeColor = UIColor(red: 0, green: 0.8, blue: 1, alpha: display.alpha * 0.8).cgColor
                lineLayer.lineWidth = 1
                layer.addSublayer(lineLayer)
                cornerLayers.append(lineLayer)
            }
            
            if !display.name.isEmpty {
                let nameLayer = CATextLayer()
                nameLayer.string = display.name
                nameLayer.fontSize = 14
                nameLayer.foregroundColor = UIColor(red: 1, green: 0.3, blue: 0, alpha: display.alpha).cgColor
                nameLayer.backgroundColor = UIColor.black.withAlphaComponent(0.5 * display.alpha).cgColor
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
    var showSettings: Binding<Bool>? = nil
    var showChat: Binding<Bool>? = nil
    
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
                
                if let chatBinding = showChat {
                    Button(action: { chatBinding.wrappedValue.toggle() }) {
                        Image(systemName: "message.fill")
                            .font(.system(size: 20))
                            .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                            .padding(.trailing, 10)
                    }
                }
                
                if let settingsBinding = showSettings {
                    Button(action: { settingsBinding.wrappedValue.toggle() }) {
                        Image(systemName: "gearshape")
                            .font(.system(size: 20))
                            .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                            .padding(.trailing, 10)
                    }
                }
                
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

struct SettingsView: View {
    @Binding var showSettings: Bool
    @State private var apiKey: String = UserDefaults.standard.string(forKey: "openrouter_api_key") ?? ""
    @State private var jarvisEnabled: Bool = UserDefaults.standard.bool(forKey: "jarvis_enabled")
    @State private var showSavedAlert = false
    @State private var showTestImagePicker = false
    @State private var testResult = ""
    @State private var isTesting = false
    @State private var isAnalyzingScreen = false
    @State private var screenAnalysisResult = ""
    
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Button(action: { showSettings = false }) {
                    Image(systemName: "xmark")
                        .font(.system(size: 24))
                        .foregroundColor(.white)
                        .padding()
                }
                Spacer()
                Text("SETTINGS")
                    .font(.system(size: 18, weight: .bold, design: .monospaced))
                    .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                Spacer()
                Button(action: saveSettings) {
                    Text("SAVE")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundColor(.green)
                        .padding()
                }
            }
            .padding(.top, 60)
            .background(Color.black.opacity(0.95))
            
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("OPENROUTER API KEY")
                            .font(.system(size: 12, weight: .bold, design: .monospaced))
                            .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                        
                        SecureField("sk-or-...", text: $apiKey)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .font(.system(size: 14))
                    }
                    .padding()
                    .background(Color.white.opacity(0.05))
                    .cornerRadius(10)
                    
                    Toggle(isOn: $jarvisEnabled) {
                        VStack(alignment: .leading, spacing: 5) {
                            Text("JARVIS VOICE")
                                .font(.system(size: 14, weight: .bold))
                                .foregroundColor(.white)
                            Text("Voice responses when faces are recognized")
                                .font(.system(size: 12))
                                .foregroundColor(.gray)
                        }
                    }
                    .padding()
                    .background(Color.white.opacity(0.05))
                    .cornerRadius(10)
                    
                    VStack(alignment: .leading, spacing: 10) {
                        Text("JARVIS SETTINGS")
                            .font(.system(size: 12, weight: .bold, design: .monospaced))
                            .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                        
                        Button(action: { Jarvis.shared.speak("Jarvis online, sir") }) {
                            HStack {
                                Image(systemName: "speaker.wave.2")
                                Text("Test Voice")
                            }
                            .padding()
                            .background(Color(red: 0.0, green: 0.6, blue: 0.8))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                        }
                        
                        Button(action: { Jarvis.shared.stop() }) {
                            HStack {
                                Image(systemName: "stop.fill")
                                Text("Stop")
                            }
                            .padding()
                            .background(Color.red.opacity(0.7))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                        }
                    }
                    .padding()
                    .background(Color.white.opacity(0.05))
                    .cornerRadius(10)
                    
                    VStack(alignment: .leading, spacing: 10) {
                        Text("MODEL")
                            .font(.system(size: 12, weight: .bold, design: .monospaced))
                            .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                        
                        Text("Using: anthropic/claude-3.5-haiku")
                            .font(.system(size: 14))
                            .foregroundColor(.gray)
                    }
                    .padding()
                    .background(Color.white.opacity(0.05))
                    .cornerRadius(10)
                    
                    VStack(alignment: .leading, spacing: 10) {
                        Text("TEST AI")
                            .font(.system(size: 12, weight: .bold, design: .monospaced))
                            .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                        
                        Button(action: { showTestImagePicker = true }) {
                            HStack {
                                Image(systemName: "photo.badge.plus")
                                Text("Test Face Recognition")
                            }
                            .padding()
                            .background(Color.orange)
                            .foregroundColor(.white)
                            .cornerRadius(8)
                        }
                        
                        if testResult != "" {
                            Text("Result: \(testResult)")
                                .font(.system(size: 12))
                                .foregroundColor(.green)
                                .padding(.top, 5)
                        }
                    }
                    .padding()
                    .background(Color.white.opacity(0.05))
                    .cornerRadius(10)
                    
                    VStack(alignment: .leading, spacing: 10) {
                        Text("SCREEN ANALYSIS")
                            .font(.system(size: 12, weight: .bold, design: .monospaced))
                            .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                        
                        Button(action: analyzeScreen) {
                            HStack {
                                Image(systemName: "rectangle.on.rectangle")
                                Text(isAnalyzingScreen ? "Analyzing..." : "What's on my screen?")
                            }
                            .padding()
                            .background(isAnalyzingScreen ? Color.gray : Color(red: 0.0, green: 0.8, blue: 1.0))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                        }
                        .disabled(isAnalyzingScreen)
                        
                        if screenAnalysisResult != "" {
                            Text(screenAnalysisResult)
                                .font(.system(size: 12))
                                .foregroundColor(.green)
                                .padding(.top, 5)
                        }
                        
                        Text("Requires JARVISScreenTweak installed")
                            .font(.system(size: 10))
                            .foregroundColor(.gray)
                    }
                    .padding()
                    .background(Color.white.opacity(0.05))
                    .cornerRadius(10)
                }
                .padding()
            }
        }
        .background(Color.black.opacity(0.95))
        .alert("Settings Saved", isPresented: $showSavedAlert) {
            Button("OK", role: .cancel) {}
        }
        .sheet(isPresented: $showTestImagePicker) {
            TestImagePicker(result: $testResult, isTesting: $isTesting)
        }
    }
    
    func saveSettings() {
        UserDefaults.standard.set(apiKey, forKey: "openrouter_api_key")
        UserDefaults.standard.set(jarvisEnabled, forKey: "jarvis_enabled")
        showSavedAlert = true
    }
    
    func analyzeScreen() {
        isAnalyzingScreen = true
        screenAnalysisResult = ""
        
        OpenRouterClient().analyzeScreen { result in
            isAnalyzingScreen = false
            screenAnalysisResult = result
        }
    }
}

struct TestImagePicker: UIViewControllerRepresentable {
    @Binding var result: String
    @Binding var isTesting: Bool
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: TestImagePicker
        
        init(_ parent: TestImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            parent.presentationMode.wrappedValue.dismiss()
            
            if let image = info[.originalImage] as? UIImage {
                parent.isTesting = true
                parent.result = "Scanning face..."
                
                if let cgImage = image.cgImage {
                    let request = VNDetectFaceRectanglesRequest()
                    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
                    try? handler.perform([request])
                    
                    if let observation = request.results?.first {
                        let classifier = FaceClassifier()
                        classifier.load()
                        
                        if let name = classifier.recognize(cgImage: cgImage) {
                            parent.result = "Hi \(name)! 👋"
                            Jarvis.shared.speak("Hello, \(name)")
                            parent.isTesting = false
                            return
                        }
                    }
                }
                
                OpenRouterClient().identifyFaceWithResponse(image: image, knownNames: []) { response in
                    DispatchQueue.main.async {
                        self.parent.result = response
                        self.parent.isTesting = false
                    }
                }
            }
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

class Jarvis: ObservableObject {
    static let shared = Jarvis()
    
    private let synthesizer = AVSpeechSynthesizer()
    private var lastSpokenTime: Date = .distantPast
    private let cooldown: TimeInterval = 3.0
    
    private init() {}
    
    func speak(_ text: String) {
        guard UserDefaults.standard.bool(forKey: "jarvis_enabled") else { return }
        guard Date().timeIntervalSince(lastSpokenTime) > cooldown else { return }
        
        lastSpokenTime = Date()
        
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-GB")
        utterance.rate = 0.52
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0
        
        synthesizer.speak(utterance)
    }
    
    func announceFaceRecognized(name: String) {
        speak("Hello, \(name)")
    }
    
    func announceTargetAcquired() {
        speak("Target acquired")
    }
    
    func announceNewFace() {
        speak("Unknown contact detected")
    }
    
    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }
}

class OpenRouterClient {
    private var apiKey: String {
        UserDefaults.standard.string(forKey: "openrouter_api_key") ?? ""
    }
    
    private var isEnabled: Bool {
        !apiKey.isEmpty
    }
    
    func identifyFace(image: UIImage, knownNames: [String], completion: @escaping (String?) -> Void) {
        guard isEnabled else {
            completion(nil)
            return
        }
        
        guard let imageData = image.jpegData(compressionQuality: 0.5) else {
            completion(nil)
            return
        }
        let base64Image = imageData.base64EncodedString()
        
        let namesList = knownNames.joined(separator: ", ")
        let prompt = """
        You are JARVIS, Tony Stark's AI assistant. Identify the person in this image.
        Known people: \(namesList.isEmpty ? "None" : namesList)
        Respond ONLY with the person's name if recognized, or "Unknown" if not recognized.
        Be brief and precise.
        """
        
        let requestBody: [String: Any] = [
            "model": "anthropic/claude-3.5-haiku",
            "messages": [
                [
                    "role": "user",
                    "content": [
                        ["type": "text", "text": prompt],
                        ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64,\(base64Image)"]
                        ]
                    ]
                ]
            ],
            "max_tokens": 50
        ]
        
        guard let url = URL(string: "https://openrouter.ai/api/v1/chat/completions") else {
            completion(nil)
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: requestBody)
        request.timeoutInterval = 10
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let choices = json["choices"] as? [[String: Any]],
                  let message = choices.first?["message"] as? [String: Any],
                  let responseText = message["content"] as? String else {
                DispatchQueue.main.async { completion(nil) }
                return
            }
            
            let name = responseText.trimmingCharacters(in: .whitespacesAndNewlines)
            DispatchQueue.main.async {
                if name.lowercased() != "unknown" && knownNames.contains(where: { name.lowercased().contains($0.lowercased()) }) {
                    completion(knownNames.first { name.lowercased().contains($0.lowercased()) })
                } else {
                    completion(nil)
                }
            }
        }.resume()
    }
    
    func identifyFaceWithResponse(image: UIImage, knownNames: [String], completion: @escaping (String) -> Void) {
        guard isEnabled else {
            completion("API key not set")
            return
        }
        
        guard let imageData = image.jpegData(compressionQuality: 0.3) else {
            completion("Failed to process image")
            return
        }
        let base64Image = imageData.base64EncodedString()
        
        let namesList = knownNames.joined(separator: ", ")
        let prompt = """
        You are JARVIS, Tony Stark's AI assistant. Analyze this face image.
        Known people: \(namesList.isEmpty ? "None yet" : namesList)
        Describe who you see and if they match any known person. Be brief.
        """
        
        let contentArray: [[String: Any]] = [
            ["type": "text", "text": prompt]
        ]
        
        let imageContent: [String: Any] = [
            "type": "image_url",
            "image_url": ["url": "data:image/jpeg;base64,\(base64Image)"]
        ]
        
        let requestBody: [String: Any] = [
            "model": "openai/gpt-4o-mini",
            "messages": [
                [
                    "role": "user",
                    "content": [imageContent, ["type": "text", "text": prompt]]
                ]
            ],
            "max_tokens": 200
        ]
        
        guard let url = URL(string: "https://openrouter.ai/api/v1/chat/completions") else {
            completion("Invalid URL")
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: requestBody)
        request.timeoutInterval = 15
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                DispatchQueue.main.async { completion("Network error: \(error.localizedDescription)") }
                return
            }
            
            guard let data = data else {
                DispatchQueue.main.async { completion("No data received") }
                return
            }
            
            if let responseString = String(data: data, encoding: .utf8) {
                print("API Response: \(responseString.prefix(500))")
            }
            
            guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                DispatchQueue.main.async { completion("Invalid JSON response") }
                return
            }
            
            if let errorMsg = json["error"] as? [String: Any], let message = errorMsg["message"] as? String {
                DispatchQueue.main.async { completion("API Error: \(message)") }
                return
            }
            
            guard let choices = json["choices"] as? [[String: Any]],
                  let message = choices.first?["message"] as? [String: Any],
                  let responseText = message["content"] as? String else {
                DispatchQueue.main.async { completion("Invalid response format") }
                return
            }
            
            DispatchQueue.main.async {
                completion(responseText.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }.resume()
    }
    
    func chat(message: String, history: [[String: String]], completion: @escaping (String) -> Void) {
        guard isEnabled else {
            completion("API key not set")
            return
        }
        
        var messages: [[String: Any]] = [
            ["role": "system", "content": "You are JARVIS, Tony Stark's AI assistant. Be helpful, witty, and British. Keep responses concise."]
        ]
        
        for msg in history {
            messages.append(["role": msg["role"] ?? "user", "content": msg["content"] ?? ""])
        }
        
        messages.append(["role": "user", "content": message])
        
        let requestBody: [String: Any] = [
            "model": "anthropic/claude-3.5-haiku",
            "messages": messages,
            "max_tokens": 300
        ]
        
        guard let url = URL(string: "https://openrouter.ai/api/v1/chat/completions") else {
            completion("Invalid URL")
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: requestBody)
        request.timeoutInterval = 30
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let choices = json["choices"] as? [[String: Any]],
                  let message = choices.first?["message"] as? [String: Any],
                  let responseText = message["content"] as? String else {
                DispatchQueue.main.async { completion("API error") }
                return
            }
            
            DispatchQueue.main.async {
                completion(responseText.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }.resume()
    }
    
    func analyzeScreen(completion: @escaping (String) -> Void) {
        guard isEnabled else {
            completion("API key not set")
            return
        }
        
        let screenshotsDir = "/var/mobile/Library/JARVIS/screenshots"
        let contextPath = "/var/mobile/Library/JARVIS/context/current.txt"
        
        guard let screenshots = try? FileManager.default.contentsOfDirectory(atPath: screenshotsDir) as [String],
              let latestFile = screenshots.sorted().last else {
            completion("No screen captures found")
            return
        }
        
        let fullPath = (screenshotsDir as NSString).appendingPathComponent(latestFile)
        guard let imageData = try? Data(contentsOf: URL(fileURLWithPath: fullPath)) else {
            completion("Could not read screenshot")
            return
        }
        
        var contextInfo = ""
        if let contextData = try? Data(contentsOf: URL(fileURLWithPath: contextPath)),
           let context = try? JSONSerialization.jsonObject(with: contextData) as? [String: Any] {
            if let app = context["app"] as? String {
                contextInfo = "The user is currently using the app: \(app)"
            }
        }
        
        let requestBody: [String: Any] = [
            "model": "anthropic/claude-3-haiku-20240307",
            "messages": [
                ["role": "system", "content": "You are JARVIS, Tony Stark's AI assistant. Analyze the user's screen and provide helpful commentary about what you see. Be witty, British, and concise."],
                ["role": "user", "content": "type: text, text: \(contextInfo) What do you see on the screen? Provide a brief analysis."]
            ],
            "max_tokens": 150
        ]
        
        guard let url = URL(string: "https://openrouter.ai/api/v1/chat/completions") else {
            completion("Invalid URL")
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: requestBody)
        request.timeoutInterval = 30
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let choices = json["choices"] as? [[String: Any]],
                  let message = choices.first?["message"] as? [String: Any],
                  let responseText = message["content"] as? String else {
                DispatchQueue.main.async { completion("Could not analyze screen") }
                return
            }
            
            DispatchQueue.main.async {
                completion(responseText.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }.resume()
    }
}

struct ChatView: View {
    @Binding var showChat: Bool
    @State private var messageText = ""
    @State private var messages: [ChatMessage] = []
    @State private var isLoading = false
    
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Button(action: { showChat = false }) {
                    Image(systemName: "xmark")
                        .font(.system(size: 24))
                        .foregroundColor(.white)
                        .padding()
                }
                Spacer()
                Text("JARVIS")
                    .font(.system(size: 18, weight: .bold, design: .monospaced))
                    .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                Spacer()
                Button(action: { messages = [] }) {
                    Image(systemName: "trash")
                        .font(.system(size: 20))
                        .foregroundColor(.red)
                        .padding()
                }
            }
            .padding(.top, 60)
            .background(Color.black.opacity(0.95))
            
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(messages) { msg in
                            ChatBubble(message: msg)
                                .id(msg.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: messages.count) { _ in
                    if let lastMessage = messages.last {
                        withAnimation {
                            proxy.scrollTo(lastMessage.id, anchor: .bottom)
                        }
                    }
                }
            }
            
            HStack(spacing: 12) {
                TextField("Talk to JARVIS...", text: $messageText)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .font(.system(size: 16))
                
                Button(action: sendMessage) {
                    if isLoading {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    } else {
                        Image(systemName: "paperplane.fill")
                            .foregroundColor(.white)
                    }
                }
                .frame(width: 44, height: 44)
                .background(messageText.isEmpty ? Color.gray : Color(red: 0.0, green: 0.8, blue: 1.0))
                .cornerRadius(22)
                .disabled(messageText.isEmpty || isLoading)
            }
            .padding()
            .background(Color.black.opacity(0.95))
        }
        .background(Color.black)
    }
    
    func sendMessage() {
        let text = messageText
        messageText = ""
        
        let userMessage = ChatMessage(role: "user", content: text)
        messages.append(userMessage)
        
        isLoading = true
        
        let history = messages.dropLast().map { ["role": $0.role, "content": $0.content] }
        
        OpenRouterClient().chat(message: text, history: Array(history)) { response in
            isLoading = false
            let jarvisMessage = ChatMessage(role: "assistant", content: response)
            messages.append(jarvisMessage)
            
            if UserDefaults.standard.bool(forKey: "jarvis_enabled") {
                Jarvis.shared.speak(response)
            }
        }
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: String
    let content: String
}

struct ChatBubble: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.role == "assistant" {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                Text("JARVIS")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
            }
            
            Text(message.content)
                .font(.system(size: 14))
                .foregroundColor(message.role == "user" ? .white : .white)
                .padding(12)
                .background(message.role == "user" ? Color(red: 0.0, green: 0.6, blue: 0.8) : Color.white.opacity(0.1))
                .cornerRadius(16)
            
            if message.role == "user" {
                Spacer()
            }
        }
    }
}
