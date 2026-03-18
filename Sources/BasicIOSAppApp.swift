import SwiftUI
import AVFoundation

struct CameraView: UIViewRepresentable {
    func makeUIView(context: Context) -> CameraPreviewView {
        let view = CameraPreviewView()
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
        view.previewLayer = previewLayer
        view.captureSession = captureSession
        
        DispatchQueue.global(qos: .userInitiated).async {
            captureSession.startRunning()
        }
        
        return view
    }
    
    func updateUIView(_ uiView: CameraPreviewView, context: Context) {}
    
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

class CameraPreviewView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate {
    var previewLayer: AVCaptureVideoPreviewLayer?
    var captureSession: AVCaptureSession?
    private var videoOutput: AVCaptureVideoDataOutput?
    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }
    
    override func didMoveToWindow() {
        super.didMoveToWindow()
        if window != nil {
            updateOrientation()
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
