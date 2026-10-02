// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "GaiaMLXQuant",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "GaiaMLXQuant", targets: ["GaiaMLXQuant"]),
        .library(name: "GaiaMIDI", targets: ["GaiaMIDI"]),
    ],
    targets: [
        // The OS-agnostic lane: pure Swift, no Foundation, no Apple frameworks.
        .target(name: "GaiaMLXQuant"),
        // The macOS lane: CoreMIDI, the OpenXTalk-Apple-CoreMIDI surface ported.
        .target(name: "GaiaMIDI"),
        .testTarget(name: "GaiaMLXQuantTests", dependencies: ["GaiaMLXQuant"]),
        .testTarget(name: "GaiaMIDITests", dependencies: ["GaiaMIDI"]),
    ]
)
