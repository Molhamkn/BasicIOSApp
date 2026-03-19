import SwiftUI
import AVFoundation
import Vision
import PhotosUI

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
    private var isTracking = false
    private var trackedObservations: [VNDetectedObjectObservation] = []
    private var frameCount = 0
    private let detectEveryNFrames = 5
    
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
        isTracking = false
        trackedObservations = []
        
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
        
        if frameCount % detectEveryNFrames == 0 || !isTracking || trackedObservations.isEmpty {
            let detectRequest = VNDetectFaceRectanglesRequest { [weak self] request, error in
                guard error == nil else { return }
                guard let results = request.results as? [VNFaceObservation] else {
                    self?.isTracking = false
                    self?.trackedObservations = []
                    return
                }
                
                if !results.isEmpty {
                    self?.trackedObservations = results.map { $0 }
                    self?.isTracking = true
                } else {
                    self?.isTracking = false
                }
            }
            
            try? handler.perform([detectRequest], on: pixelBuffer)
        } else if isTracking && !trackedObservations.isEmpty {
            let trackRequest = VNTrackObjectRequest(detectedObjectObservation: trackedObservations.first!)
            try? handler.perform([trackRequest], on: pixelBuffer)
            
            if let result = trackRequest.results?.first as? VNDetectedObjectObservation {
                trackedObservations = [result]
            } else {
                isTracking = false
            }
        }
        
        let faceEncoding = extractEncoding(from: pixelBuffer)
        
        var faceTargets = trackedObservations.map { observation -> FaceTarget in
            var target = FaceTarget(rect: observation.boundingBox)
            if let encoding = faceEncoding, let name = faceClassifier?.recognize(encoding: encoding) {
                target.recognizedName = name
            }
            return target
        }
        
        onFacesDetected?(faceTargets)
    }
    
    private func extractEncoding(from pixelBuffer: CVPixelBuffer) -> Data? {
        let request = VNGenerateImageFeaturePrintRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        try? handler.perform([request])
        
        guard let result = request.results?.first as? VNFeaturePrintObservation else { return nil }
        
        var data = Data()
        for i in 0..<128 {
            var value: Float = 0
            try? result.featurePrint(at: i, value: &value)
            withUnsafeBytes(of: value) { data.append(contentsOf: $0) }
        }
        
        return data
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processFrame(pixelBuffer)
    }
}

class FaceClassifier {
    struct FaceEncoding {
        var data: Data
        var label: String
    }
    
    private var encodings: [FaceEncoding] = []
    private var isTrained = false
    
    func addEncoding(_ data: Data, label: String) {
        encodings.append(FaceEncoding(data: data, label: label))
        isTrained = true
    }
    
    func recognize(encoding: Data) -> String? {
        guard isTrained, !encodings.isEmpty else { return nil }
        
        var bestMatch: (label: String, distance: Float) = (label: "", distance: Float.greatestFiniteMagnitude)
        
        for sample in encodings {
            let distance = compareEncodings(encoding, sample.data)
            if distance < bestMatch.distance && distance < 0.6 {
                bestMatch = (label: sample.label, distance: distance)
            }
        }
        
        if bestMatch.distance < 0.6 {
            return bestMatch.label
        }
        return nil
    }
    
    private func compareEncodings(_ a: Data, _ b: Data) -> Float {
        guard a.count == b.count else { return Float.greatestFiniteMagnitude }
        
        var sum: Float = 0
        let aBytes = [UInt8](a)
        let bBytes = [UInt8](b)
        
        for i in 0..<min(aBytes.count, bBytes.count) {
            let diff = Float(aBytes[i]) - Float(bBytes[i])
            sum += diff * diff
        }
        
        return sqrt(sum) / Float(aBytes.count)
    }
    
    func getTrainedNames() -> [String] {
        return Array(Set(encodings.map { $0.label })).sorted()
    }
    
    func save() {
        let data = encodings.map { ["data": $0.data, "label": $0.label] }
        if let encoded = try? NSKeyedArchiver.archivedData(withRootObject: data, requiringSecureCoding: false) {
            UserDefaults.standard.set(encoded, forKey: "FaceEncodings")
        }
    }
    
    func load() {
        guard let data = UserDefaults.standard.data(forKey: "FaceEncodings"),
              let decoded = try? NSKeyedUnarchiver.unarchivedArrayOfObjects(ofClass: NSDictionary.self, from: data) as? [[String: Any]] else { return }
        encodings = decoded.compactMap { dict in
            guard let data = dict["data"] as? Data, let label = dict["label"] as? String else { return nil }
            return FaceEncoding(data: data, label: label)
        }
        isTrained = !encodings.isEmpty
    }
    
    func clear() {
        encodings = []
        isTrained = false
        UserDefaults.standard.removeObject(forKey: "FaceEncodings")
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
            
            Button(action: { showImagePicker = true }) {
                HStack {
                    Image(systemName: "photo.on.rectangle.angled")
                    Text("Select Photos")
                }
                .padding()
                .background(Color(red: 0.0, green: 0.6, blue: 0.8))
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .padding()
            
            if !capturedImages.isEmpty {
                Text("\(capturedImages.count) photos selected")
                    .foregroundColor(.green)
                    .padding(.vertical, 5)
                
                HStack {
                    ForEach(0..<min(capturedImages.count, 4), id: \.self) { i in
                        Image(uiImage: capturedImages[i])
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(width: 60, height: 60)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
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
                    ForEach(trainedPeople, id: \.self) { name in
                        Text("• \(name)")
                            .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
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
                if let encoding = extractFaceEncoding(from: image) {
                    cameraManager.faceClassifier?.addEncoding(encoding, label: newPersonName)
                }
            }
            
            cameraManager.faceClassifier?.save()
            
            DispatchQueue.main.async {
                isTraining = false
                trainingComplete = true
                capturedImages = []
                trainedPeople = cameraManager.faceClassifier?.getTrainedNames() ?? []
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    trainingComplete = false
                }
            }
        }
    }
    
    func extractFaceEncoding(from image: UIImage) -> Data? {
        guard let cgImage = image.cgImage else { return nil }
        
        let request = VNGenerateImageFeaturePrintRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        try? handler.perform([request])
        
        guard let result = request.results?.first as? VNFeaturePrintObservation else { return nil }
        
        var data = Data()
        for i in 0..<128 {
            var value: Float = 0
            try? result.featurePrint(at: i, value: &value)
            withUnsafeBytes(of: value) { data.append(contentsOf: $0) }
        }
        
        return data
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
        let parent: ImagePicker
        
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
    let showCameraSwitcher: Binding<Bool>
    
    var body: some View {
        Button(action: {
            cameraManager.switchCamera()
            showCameraSwitcher.wrappedValue = false
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
