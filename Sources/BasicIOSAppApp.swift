import SwiftUI
import AVFoundation

struct CameraView: UIViewRepresentable {
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
        
        let pinchGesture = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePinch(_:)))
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
        Coordinator()
    }
    
    class Coordinator: NSObject {
        var captureSession: AVCaptureSession?
        var previewLayer: AVCaptureVideoPreviewLayer?
        var camera: AVCaptureDevice?
        var currentZoom: CGFloat = 1.0
        
        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            guard let camera = camera else { return }
            
            do {
                try camera.lockForConfiguration()
                let maxZoom = min(camera.activeFormat.videoMaxZoomFactor, 10.0)
                let minZoom: CGFloat = 1.0
                
                if gesture.state == .began {
                    currentZoom = camera.videoZoomFactor
                } else if gesture.state == .changed {
                    let newZoom = currentZoom * gesture.scale
                    camera.videoZoomFactor = max(minZoom, min(newZoom, maxZoom))
                }
                
                camera.unlockForConfiguration()
            } catch {}
        }
        
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
