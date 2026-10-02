// CoreMIDILane.swift — the macOS CoreMIDI lane.
//
// A Swift port of the surface exposed by the OpenXTalk-Apple-CoreMIDI
// Builder library (MIT, Paul McClernan): register a MIDI client, register a
// virtual source, send Note On/Off and raw bytes with timestamp conversion,
// and enumerate the current MIDI setup. The OpenXTalk stack scripts reach
// the same CoreMIDI services through LCB; this lane reaches them from the
// alexiai Swift package, so the music lane can drive MIDI from any of the
// constellation's languages.
//
// macOS only — guarded by `#if canImport(CoreMIDI)` so the package still
// builds on Linux (the GaiaMLXQuant core is OS-agnostic; this lane simply
// does not exist there).

#if canImport(CoreMIDI)
import CoreMIDI
import Darwin

/// Utilities shared by the client and the tests. All pure — no devices
/// needed.
public enum GaiaMIDIMessage {

    /// Status byte for note-on on a channel (0-15).
    @inline(__always)
    public static func noteOn(_ note: UInt8, channel: UInt8 = 0, velocity: UInt8 = 100) -> [UInt8] {
        [0x90 | (channel & 0x0F), note & 0x7F, velocity & 0x7F]
    }

    /// Status byte for note-off on a channel (0-15).
    @inline(__always)
    public static func noteOff(_ note: UInt8, channel: UInt8 = 0, velocity: UInt8 = 64) -> [UInt8] {
        [0x80 | (channel & 0x0F), note & 0x7F, velocity & 0x7F]
    }

    /// A control-change message.
    @inline(__always)
    public static func controlChange(_ controller: UInt8, value: UInt8, channel: UInt8 = 0) -> [UInt8] {
        [0xB0 | (channel & 0x0F), controller & 0x7F, value & 0x7F]
    }

    /// Variable-length quantity encode (Standard MIDI File style, 7 bits per
    /// byte, most significant first). The same VLQ the OpenXTalk library's
    /// TODO list promises for timestamp encoding.
    public static func vlqEncode(_ value: UInt32) -> [UInt8] {
        var buffer: [UInt8] = [UInt8(value & 0x7F)]
        var v = value >> 7
        while v > 0 {
            buffer.insert(UInt8(v & 0x7F) | 0x80, at: 0)
            v >>= 7
        }
        return buffer
    }

    /// Variable-length quantity decode.
    public static func vlqDecode(_ bytes: [UInt8]) -> UInt32 {
        var value: UInt32 = 0
        for byte in bytes {
            value = (value << 7) | UInt32(byte & 0x7F)
        }
        return value
    }

    /// CoreMIDI host time (mach_absolute_time) to milliseconds, using the
    /// same timebase CoreMIDI uses — the division values the OpenXTalk
    /// library exposes to script as numerator/denominator.
    public static func hostTimeToMilliseconds(_ hostTime: UInt64) -> Double {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return Double(hostTime) * Double(info.numer) / Double(info.denom) / 1_000_000.0
    }
}

/// A MIDI client and virtual source, mirroring the OpenXTalk library's
/// `MIDIClientCreate` + `MIDISourceCreate` + send-bytes flow.
public final class GaiaMIDIClient {
    public let name: String
    private var clientRef = MIDIClientRef()

    public init(name: String) throws {
        self.name = name
        let status = MIDIClientCreateWithBlock(name as CFString, &clientRef) { _ in
            // Setup-change notification: the OpenXTalk library surfaces this
            // to script; here it is a no-op hook for future wiring.
        }
        guard status == noErr else {
            throw GaiaMIDIError.clientCreateFailed(status)
        }
    }

    deinit {
        MIDIClientDispose(clientRef)
    }

    /// The sources and destinations CoreMIDI currently sees.
    public func listSources() -> [String] {
        names(for: MIDIGetNumberOfSources(), source: MIDIGetSource)
    }

    public func listDestinations() -> [String] {
        names(for: MIDIGetNumberOfDestinations(), source: MIDIGetDestination)
    }

    private func names(
        for count: Int,
        source: (Int) -> MIDIEndpointRef
    ) -> [String] {
        (0..<count).compactMap { i in
            let endpoint = source(i)
            var name: Unmanaged<CFString>?
            let status = MIDIObjectGetStringProperty(endpoint, kMIDIPropertyName, &name)
            guard status == noErr else { return nil }
            return name?.takeRetainedValue() as String?
        }
    }

    /// Register a virtual source this client can send through — the library's
    /// headline feature.
    public func makeVirtualSource(_ name: String) throws -> GaiaVirtualSource {
        let source = try GaiaVirtualSource(client: clientRef, name: name)
        return source
    }
}

/// A virtual MIDI source; `send` pushes bytes into the MIDI system.
public final class GaiaVirtualSource {
    public let name: String
    private var endpointRef = MIDIEndpointRef()

    init(client: MIDIClientRef, name: String) throws {
        self.name = name
        var status = MIDISourceCreate(client, name as CFString, &endpointRef)
        guard status == noErr else {
            throw GaiaMIDIError.sourceCreateFailed(status)
        }
    }

    deinit {
        MIDIEndpointDispose(endpointRef)
    }

    /// Send raw MIDI bytes with a timestamp now.
    public func send(_ bytes: [UInt8], timestampMs: Double? = nil) throws {
        let hostTime: MIDITimeStamp = if let timestampMs {
            MIDITimeStamp(timestampMs * 1_000_000.0 * timebaseRatio())
        } else {
            mach_absolute_time()
        }
        var packet = MIDIPacket()
        packet.timeStamp = hostTime
        packet.length = UInt16(bytes.count)
        bytes.withUnsafeBufferPointer { buffer in
            buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: bytes.count) { p in
                withUnsafeMutableBytes(of: &packet.data) { dst in
                    dst.baseAddress!.copyMemory(from: p, byteCount: bytes.count)
                }
            }
        }
        var packetList = MIDIPacketList(numPackets: 1, packet: packet)
        let status = MIDIReceived(endpointRef, &packetList)
        guard status == noErr else {
            throw GaiaMIDIError.sendFailed(status)
        }
    }

    /// Convenience: play a note for `durationMs` (blocking, millisecond-scale
    /// sleep — this is a demo lane, not a sequencer).
    public func playNote(
        _ note: UInt8,
        channel: UInt8 = 0,
        velocity: UInt8 = 100,
        durationMs: Double = 200
    ) throws {
        try send(GaiaMIDIMessage.noteOn(note, channel: channel, velocity: velocity))
        usleep(useconds_t(durationMs * 1000))
        try send(GaiaMIDIMessage.noteOff(note, channel: channel))
    }

    private func timebaseRatio() -> Double {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return Double(info.numer) / Double(info.denom)
    }
}

public enum GaiaMIDIError: Error, CustomStringConvertible {
    case clientCreateFailed(OSStatus)
    case sourceCreateFailed(OSStatus)
    case sendFailed(OSStatus)

    public var description: String {
        switch self {
        case .clientCreateFailed(let s): return "MIDIClientCreate failed: \(s)"
        case .sourceCreateFailed(let s): return "MIDISourceCreate failed: \(s)"
        case .sendFailed(let s): return "MIDIReceived failed: \(s)"
        }
    }
}
#endif
