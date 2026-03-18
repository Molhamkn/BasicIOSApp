import SwiftUI
import AVFoundation

struct CameraContainerView: View {
    @StateObject private var cameraManager = CameraManager()
    @State private var showCameraSwitcher = false
    @GestureState private var gestureZoom: CGFloat = 1.0
    
    var body: some View {
        ZStack {
            CameraPreviewViewRepresentable(cameraManager: cameraManager)
                .ignoresSafeArea()
                .gesture(
                    MagnificationGesture()
                        .updating($gestureZoom) { value, state, _ in
                            state = value
                        }
                        .onEnded { value in
                            let newZoom = cameraManager.zoom * value
                            cameraManager.setZoom(newZoom)
                        }
                )
            
            IronManHUD(
                currentZoom: cameraManager.zoom,
                showCameraSwitcher: $showCameraSwitcher
            )
        }
        .onAppear {
            cameraManager.setup()
        }
    }
}

class CameraManager: NSObject, ObservableObject {
    @Published var zoom: CGFloat = 1.0
    @Published var isFrontCamera: Bool = false
    @Published var isReady: Bool = false
    
    let captureSession = AVCaptureSession()
    var currentInput: AVCaptureDeviceInput?
    
    func setup() {
        setupCamera(position: .back)
    }
    
    func setupCamera(position: AVCaptureDevice.Position) {
        captureSession.beginConfiguration()
        captureSession.inputs.forEach { captureSession.removeInput($0) }
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            captureSession.commitConfiguration()
            return
        }
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
            currentInput = input
        }
        
        captureSession.commitConfiguration()
        isFrontCamera = (position == .front)
        isReady = true
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
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
        guard let input = currentInput,
              let device = input.device else { return }
        
        let maxZoom = min(device.activeFormat.videoMaxZoomFactor, 10.0)
        let minZoom: CGFloat = 1.0
        zoom = max(minZoom, min(newZoom, maxZoom))
        
        do {
            try device.lockForConfiguration()
            device.videoZoomFactor = zoom
            device.unlockForConfiguration()
        } catch {}
    }
}

struct CameraPreviewViewRepresentable: UIViewRepresentable {
    @ObservedObject var cameraManager: CameraManager
    
    func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        view.backgroundColor = .black
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: cameraManager.captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.connection?.videoOrientation = .landscapeRight
        view.previewLayer = previewLayer
        view.cameraManager = cameraManager
        
        return view
    }
    
    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {
        uiView.previewLayer?.frame = uiView.bounds
    }
}

class CameraPreviewUIView: UIView {
    var previewLayer: AVCaptureVideoPreviewLayer?
    weak var cameraManager: CameraManager?
    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }
}

struct IronManHUD: View {
    let currentZoom: CGFloat
    @Binding var showCameraSwitcher: Bool
    
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
                CameraSwitchButton {
                    showCameraSwitcher.toggle()
                }
                .padding(.leading, 20)
                .padding(.bottom, 20)
                
                Spacer()
                
                if showCameraSwitcher {
                    CameraSwitchMenu(isFront: false)
                        .padding(.trailing, 20)
                        .padding(.bottom, 20)
                }
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
        formatter.dateFormat = "HH:mm:ss"
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
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Image(systemName: "camera.rotate")
                .font(.system(size: 24))
                .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
                .shadow(color: Color(red: 0.0, green: 0.5, blue: 1.0), radius: 5)
                .frame(width: 50, height: 50)
        }
    }
}

struct CameraSwitchMenu: View {
    let isFront: Bool
    
    var body: some View {
        Text("Camera switch placeholder")
            .font(.system(size: 14, weight: .bold, design: .monospaced))
            .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
            .padding()
            .background(Color.black.opacity(0.5))
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
