import SwiftUI
import AVFoundation

struct CameraView: UIViewRepresentable {
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        let captureSession = AVCaptureSession()
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            return view
        }
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = UIScreen.main.bounds
        view.layer.addSublayer(previewLayer)
        
        context.coordinator.captureSession = captureSession
        
        DispatchQueue.global(qos: .userInitiated).async {
            captureSession.startRunning()
        }
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        if let previewLayer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            previewLayer.frame = UIScreen.main.bounds
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject {
        var captureSession: AVCaptureSession?
        
        deinit {
            captureSession?.stopRunning()
        }
    }
}

@main
struct BasicIOSAppApp: App {
    var body: some Scene {
        WindowGroup {
            CameraView()
                .ignoresSafeArea()
        }
    }
}
