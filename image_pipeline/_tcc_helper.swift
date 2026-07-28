import AVFoundation
import Foundation

let sem = DispatchSemaphore(value: 0)
var result = "unknown"

let status = AVCaptureDevice.authorizationStatus(for: .video)
switch status {
case .authorized:
    result = "allowed"
    sem.signal()
case .denied:
    result = "denied"
    sem.signal()
case .restricted:
    result = "restricted"
    sem.signal()
case .notDetermined:
    print("notDetermined", terminator: "")
    fflush(stdout)
    AVCaptureDevice.requestAccess(for: .video) { granted in
        result = granted ? "triggered_allowed" : "triggered_denied"
        sem.signal()
    }
@unknown default:
    result = "unknown_status"
    sem.signal()
}

_ = sem.wait(timeout: .now() + 30)
print(result)
