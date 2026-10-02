#if canImport(CoreMIDI)
import XCTest
@testable import GaiaMIDI

final class GaiaMIDITests: XCTestCase {

    func testNoteMessages() {
        XCTAssertEqual(GaiaMIDIMessage.noteOn(60), [0x90, 60, 100])
        XCTAssertEqual(GaiaMIDIMessage.noteOn(60, channel: 3, velocity: 7), [0x93, 60, 7])
        XCTAssertEqual(GaiaMIDIMessage.noteOff(60), [0x80, 60, 64])
        XCTAssertEqual(GaiaMIDIMessage.controlChange(1, value: 2), [0xB0, 1, 2])
    }

    func testVlqRoundTrip() {
        // The values the Standard MIDI File spec uses to pin VLQ.
        for value: UInt32 in [0, 0x40, 0x7F, 0x80, 0x2000, 0x3FFF, 0x4000, 0x0FFFFFFF] {
            let encoded = GaiaMIDIMessage.vlqEncode(value)
            XCTAssertEqual(GaiaMIDIMessage.vlqDecode(encoded), value, "value \(value)")
        }
        // The spec's canonical example: 0x80 0x00 encodes 0.
        XCTAssertEqual(GaiaMIDIMessage.vlqDecode([0x80, 0x00]), 0)
    }

    func testHostTimeIsMonotonicMilliseconds() {
        let t0 = GaiaMIDIMessage.hostTimeToMilliseconds(mach_absolute_time())
        usleep(50_000)
        let t1 = GaiaMIDIMessage.hostTimeToMilliseconds(mach_absolute_time())
        // ~50ms later, within generous bounds, and strictly increasing.
        let delta = t1 - t0
        XCTAssertGreaterThan(delta, 30)
        XCTAssertLessThan(delta, 90)
    }

    func testClientAndVirtualSourceCreateAndSend() throws {
        // Headless-safe: a client and a virtual source need no hardware.
        let client = try GaiaMIDIClient(name: "alexiai-test")
        let source = try client.makeVirtualSource("alexiai-test-source")
        try source.send(GaiaMIDIMessage.noteOn(64))
        try source.send(GaiaMIDIMessage.noteOff(64))
        // Enumeration must not crash even with zero devices.
        _ = client.listSources()
        _ = client.listDestinations()
    }
}
#endif
