import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack {
            Text("Hello iOS!")
                .font(.largeTitle)
                .padding()
            
            Text("Built with GitHub Actions")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
}

#Preview {
    ContentView()
}
