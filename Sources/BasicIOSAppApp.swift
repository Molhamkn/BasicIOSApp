import SwiftUI
import AVFoundation
import Vision

struct CameraContainerView: View {
    @StateObject private var cameraManager = CameraManager()
    @State private var showCameraSwitcher = false
    @State private var baseZoom: CGFloat = 1.0
    @State private var detectedFaces: [FaceTarget] = []
    
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
                faceCount: detectedFaces.count
            )
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
}

class CameraManager: NSObject, ObservableObject {
    @Published var zoom: CGFloat = 1.0
    @Published var isFrontCamera: Bool = false
    @Published var isReady: Bool = false
    
    let captureSession = AVCaptureSession()
    var currentInput: AVCaptureDeviceInput?
    var videoOutput: AVCaptureVideoDataOutput?
    var onFacesDetected: (([FaceTarget]) -> Void)?
    
    private var sequenceHandler: VNSequenceRequestHandler?
    private var lastProcessedTime: TimeInterval = 0
    private let processingInterval: TimeInterval = 1.0 / 30.0
    
    func setup() {
        sequenceHandler = VNSequenceRequestHandler()
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
        
        do {
            try camera.lockForConfiguration()
            if camera.activeFormat.videoSupportedFrameRateRanges.contains(where: { $0.maxFrameRate >= 60 }) {
                camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 60)
                camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 60)
            }
            camera.unlockForConfiguration()
        } catch {}
        
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
    
    private func detectFaces(in pixelBuffer: CVPixelBuffer) {
        let currentTime = CACurrentMediaTime()
        guard currentTime - lastProcessedTime >= processingInterval else { return }
        lastProcessedTime = currentTime
        
        guard let handler = sequenceHandler else { return }
        
        let request = VNDetectFaceLandmarksRequest { [weak self] request, error in
            guard error == nil else {
                let rectRequest = VNDetectFaceRectanglesRequest { rectRequest, rectError in
                    guard rectError == nil else { return }
                    guard let results = rectRequest.results as? [VNFaceObservation] else { return }
                    let rects = results.map { $0.boundingBox }
                    self?.onFacesDetected?(rects.map { FaceTarget(rect: $0) })
                }
                try? handler.perform([rectRequest], on: pixelBuffer)
                return
            }
            
            guard let results = request.results as? [VNFaceObservation] else { return }
            
            let faceRects = results.map { observation -> CGRect in
                if let landmarks = observation.landmarks {
                    let allPoints = landmarks.allPoints
                    let normalizedPoints = allPoints.normalizedPoints
                    
                    if !normalizedPoints.isEmpty {
                        var minX: CGFloat = 1.0
                        var maxX: CGFloat = 0.0
                        var minY: CGFloat = 1.0
                        var maxY: CGFloat = 0.0
                        
                        for point in normalizedPoints {
                            minX = min(minX, point.x)
                            maxX = max(maxX, point.x)
                            minY = min(minY, point.y)
                            maxY = max(maxY, point.y)
                        }
                        
                        let padding: CGFloat = 0.1
                        let centerX = observation.boundingBox.midX
                        let centerY = observation.boundingBox.midY
                        let width = observation.boundingBox.width
                        let height = observation.boundingBox.height
                        
                        return CGRect(
                            x: centerX - width * (0.5 + padding),
                            y: centerY - height * (0.5 + padding),
                            width: width * (1 + padding * 2),
                            height: height * (1 + padding * 2)
                        )
                    }
                }
                return observation.boundingBox
            }
            
            self?.onFacesDetected?(faceRects.map { FaceTarget(rect: $0) })
        }
        
        request.preferBackgroundProcessing = false
        
        try? handler.perform([request], on: pixelBuffer)
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        detectFaces(in: pixelBuffer)
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
    
    func updateFaces(_ faces: [FaceTarget], bounds: CGRect) {
        faceLayers.forEach { $0.removeFromSuperlayer() }
        cornerLayers.forEach { $0.removeFromSuperlayer() }
        faceLayers.removeAll()
        cornerLayers.removeAll()
        
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
                (0, -innerRadius - lineLength),
                (0, innerRadius + lineLength),
                (-innerRadius - lineLength, 0),
                (innerRadius + lineLength, 0)
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
        }
    }
}

struct IronManHUD: View {
    let currentZoom: CGFloat
    @Binding var showCameraSwitcher: Bool
    let cameraManager: CameraManager
    var faceCount: Int = 0
    
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
