import SwiftUI
import AVFoundation

struct CameraView: UIViewRepresentable {
    @Binding var currentZoom: CGFloat
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .black
        
        let captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            return view
        }
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.connection?.videoOrientation = .landscapeRight
        view.layer.addSublayer(previewLayer)
        
        context.coordinator.previewLayer = previewLayer
        context.coordinator.captureSession = captureSession
        context.coordinator.camera = camera
        context.coordinator.isFrontCamera = false
        
        let pinchGesture = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePinch(_:)))
        pinchGesture.delegate = context.coordinator
        view.addGestureRecognizer(pinchGesture)
        
        DispatchQueue.global(qos: .userInitiated).async {
            captureSession.startRunning()
        }
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            uiView.layer.sublayers?.first?.frame = uiView.bounds
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(currentZoom: $currentZoom)
    }
    
    class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var captureSession: AVCaptureSession?
        var previewLayer: AVCaptureVideoPreviewLayer?
        var camera: AVCaptureDevice?
        var isFrontCamera: Bool = false
        var currentZoom: Binding<CGFloat>
        
        init(currentZoom: Binding<CGFloat>) {
            self.currentZoom = currentZoom
        }
        
        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            guard let camera = camera else { return }
            
            do {
                try camera.lockForConfiguration()
                let maxZoom = min(camera.activeFormat.videoMaxZoomFactor, 10.0)
                let minZoom: CGFloat = 1.0
                let savedZoom = currentZoom.wrappedValue
                
                if gesture.state == .began {
                } else if gesture.state == .changed {
                    let newZoom = savedZoom * gesture.scale
                    camera.videoZoomFactor = max(minZoom, min(newZoom, maxZoom))
                }
                
                camera.unlockForConfiguration()
            } catch {}
        }
        
        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
            return true
        }
        
        deinit {
            captureSession?.stopRunning()
        }
    }
}

@main
struct BasicIOSAppApp: App {
    @State private var currentZoom: CGFloat = 1.0
    @State private var showCameraSwitcher = false
    
    var body: some Scene {
        WindowGroup {
            ZStack {
                CameraView(currentZoom: $currentZoom)
                    .ignoresSafeArea()
                
                IronManHUD(currentZoom: $currentZoom, showCameraSwitcher: $showCameraSwitcher)
                
                if showCameraSwitcher {
                    CameraSwitchOverlay(isFront: false)
                        .onTapGesture {
                            showCameraSwitcher = false
                        }
                }
            }
        }
    }
}

struct IronManHUD: View {
    @Binding var currentZoom: CGFloat
    @Binding var showCameraSwitcher: Bool
    @State private var currentTime = Date()
    
    let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
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
                    }
                }
            }
        }
        .onReceive(timer) { time in
            currentTime = time
        }
    }
}

struct TimeDisplay: View {
    @State private var currentTime = Date()
    let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    
    var body: some View {
        Text(timeString)
            .font(.system(size: 24, weight: .light, design: .monospaced))
            .foregroundColor(Color(red: 0.0, green: 0.8, blue: 1.0))
            .shadow(color: Color(red: 0.0, green: 0.5, blue: 1.0), radius: 5)
    }
    
    private var timeString: String {
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

struct CameraSwitchOverlay: View {
    let isFront: Bool
    
    var body: some View {
        VStack {
            Spacer()
            
            HStack {
                Spacer()
                
                Text("SWITCHING NOT IMPLEMENTED YET")
                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                    .foregroundColor(.red)
                    .padding()
                
                Spacer()
            }
            
            Spacer()
        }
        .background(Color.black.opacity(0.5))
    }
}
