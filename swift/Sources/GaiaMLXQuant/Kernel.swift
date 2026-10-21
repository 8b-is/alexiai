// Kernel.swift — the Swift lane of the GAIA-MLX-QUANT substrate.
//
// The same ternary b1.58 math as the Rust core, the Go lane, and the C99
// lane, in pure Swift. OS-agnostic by construction: this file imports
// nothing but the Swift standard library — no Foundation, no Apple
// frameworks — so it runs on macOS and Linux alike.
//
// The bithacks survive the port:
// - 5 trits per byte, base-3 (log2(3) ≈ 1.585 bits/weight)
// - branchless digit mapping: digit = UInt8(t) + (UInt8(t) >> 7) * 3
// - prebuilt slot LUTs (no division, no modulo in the hot path)
// - the masked kernel, j-major (the JS-parity layout)

public enum GaiaMLXQuant {

    // MARK: constants

    /// Five trits per byte.
    public static let tritsPerByte = 5

    /// Base-3 place values.
    public static let pow3: [UInt8] = [1, 3, 9, 27, 81]

    /// log2(3) — the exact bits-per-weight budget.
    public static let bitsPerWeight = 1.584962500721156

    /// The five slot LUTs, built once.
    public static let slotLuts: [[Int8]] = (0..<5).map { slot in
        (0..<256).map { byte in
            let digit = (byte / Int(pow3[slot])) % 3
            return digit == 2 ? -1 : Int8(digit)
        }
    }

    // MARK: packing

    /// The branchless digit mapping: -1 → 2, 0 → 0, +1 → 1.
    @inline(__always)
    public static func digitOf(_ trit: Int8) -> UInt8 {
        UInt8(bitPattern: trit) &+ (UInt8(bitPattern: trit) >> 7) &* 3
    }

    /// Pack trits into a caller-owned, zero-initialized buffer.
    public static func packTrits(_ trits: [Int8], into out: inout [UInt8]) {
        for (i, t) in trits.enumerated() {
            out[i / tritsPerByte] &+= pow3[i % tritsPerByte] &* digitOf(t)
        }
    }

    /// Unpack bytes back into trits.
    public static func unpackTrits(_ bytes: [UInt8], count: Int, into out: inout [Int8]) {
        for (i, b) in bytes.enumerated() {
            let base = i * tritsPerByte
            if base >= count { break }
            let luts = slotLuts
            out[base] = luts[0][Int(b)]
            if base + 1 < count { out[base + 1] = luts[1][Int(b)] }
            if base + 2 < count { out[base + 2] = luts[2][Int(b)] }
            if base + 3 < count { out[base + 3] = luts[3][Int(b)] }
            if base + 4 < count { out[base + 4] = luts[4][Int(b)] }
        }
    }

    // MARK: the kernel

    /// The masked kernel, j-major:
    ///
    ///     out[i] = gamma * sum_j trit[j][i] * scale[j] * x[j]
    public static func ternaryMatMul(
        packed: [UInt8],
        rows: Int,
        cols: Int,
        x: [Double],
        scales: [Double]?,
        gamma: Double,
        into out: inout [Double]
    ) {
        for i in 0..<cols { out[i] = 0 }
        let luts = slotLuts
        for j in 0..<rows {
            let xv = x[j]
            if xv == 0 { continue }
            var xj = xv * gamma
            if let scales { xj *= scales[j] }
            let startT = j * cols
            let endT = startT + cols
            let startB = startT / tritsPerByte
            let endB = (endT + tritsPerByte - 1) / tritsPerByte
            for b in startB..<endB {
                let byte = packed[b]
                if byte == 0 { continue }
                let base = b * tritsPerByte
                let t0 = base
                if t0 >= startT && t0 < endT {
                    out[t0 - startT] += Double(luts[0][Int(byte)]) * xj
                }
                let t1 = base + 1
                if t1 >= startT && t1 < endT {
                    out[t1 - startT] += Double(luts[1][Int(byte)]) * xj
                }
                let t2 = base + 2
                if t2 >= startT && t2 < endT {
                    out[t2 - startT] += Double(luts[2][Int(byte)]) * xj
                }
                let t3 = base + 3
                if t3 >= startT && t3 < endT {
                    out[t3 - startT] += Double(luts[3][Int(byte)]) * xj
                }
                let t4 = base + 4
                if t4 >= startT && t4 < endT {
                    out[t4 - startT] += Double(luts[4][Int(byte)]) * xj
                }
            }
        }
    }

    /// The float64 reference.
    public static func denseMatMul(
        _ w: [Double],
        rows: Int,
        cols: Int,
        x: [Double],
        into out: inout [Double]
    ) {
        for i in 0..<cols { out[i] = 0 }
        for j in 0..<rows {
            let xv = x[j]
            if xv == 0 { continue }
            for i in 0..<cols {
                out[i] += w[j * cols + i] * xv
            }
        }
    }

    // MARK: quantization

    /// Per-row absmean quantization — the scheme real b1.58 models are
    /// trained into.
    public static func quantizePerRow(
        _ w: [Double],
        rows: Int,
        cols: Int
    ) -> (trits: [Int8], scales: [Double]) {
        var trits = [Int8](repeating: 0, count: w.count)
        var scales = [Double](repeating: 0, count: rows)
        for r in 0..<rows {
            var sum = 0.0
            for c in 0..<cols { sum += abs(w[r * cols + c]) }
            var scale = sum / Double(cols)
            if scale == 0 { scale = 1 }
            scales[r] = scale
            let posThresh = 0.5 * scale
            let negThresh = -posThresh
            for c in 0..<cols {
                let v = w[r * cols + c]
                trits[r * cols + c] = v >= posThresh ? 1 : (v <= negThresh ? -1 : 0)
            }
        }
        return (trits, scales)
    }

    // MARK: p/q ("Counting Divided" — integer-numerator engine)

    /// Default dyadic denominator q = 2^20 = 1,048,576.
    public static let pqDefaultQ: Int64 = 1 << 20

    /// Project a Double slice onto Int32 numerators pOut[k] = round(x[k] * q) in O(K).
    public static func quantizePqInto(_ x: [Double], q: Int64, into pOut: inout [Int32]) {
        let qf = Double(q)
        for i in x.indices {
            pOut[i] = Int32((x[i] * qf).rounded())
        }
    }

    /// Pure p/q ("Counting Divided") integer-numerator masked kernel:
    /// accumulates in Z (accScratch) with zero inner-loop Int8->Double conversions,
    /// dividing by qX only once per output column.
    public static func ternaryMatMulPq(
        packed: [UInt8],
        rows: Int,
        cols: Int,
        pX: [Int32],
        qX: Int64,
        scales: [Double]?,
        gamma: Double,
        accScratch: inout [Int64],
        into out: inout [Double]
    ) {
        for i in 0..<cols { accScratch[i] = 0 }
        let luts = slotLuts
        for j in 0..<rows {
            var pxj = Int64(pX[j])
            if pxj == 0 { continue }
            if let scales {
                pxj = Int64((Double(pxj) * scales[j]).rounded())
                if pxj == 0 { continue }
            }
            let startT = j * cols
            let endT = startT + cols
            let startB = startT / tritsPerByte
            let endB = (endT + tritsPerByte - 1) / tritsPerByte
            for b in startB..<endB {
                let byte = packed[b]
                if byte == 0 { continue }
                let base = b * tritsPerByte
                let t0 = base
                if t0 >= startT && t0 < endT {
                    accScratch[t0 - startT] &+= Int64(luts[0][Int(byte)]) &* pxj
                }
                let t1 = base + 1
                if t1 >= startT && t1 < endT {
                    accScratch[t1 - startT] &+= Int64(luts[1][Int(byte)]) &* pxj
                }
                let t2 = base + 2
                if t2 >= startT && t2 < endT {
                    accScratch[t2 - startT] &+= Int64(luts[2][Int(byte)]) &* pxj
                }
                let t3 = base + 3
                if t3 >= startT && t3 < endT {
                    accScratch[t3 - startT] &+= Int64(luts[3][Int(byte)]) &* pxj
                }
                let t4 = base + 4
                if t4 >= startT && t4 < endT {
                    accScratch[t4 - startT] &+= Int64(luts[4][Int(byte)]) &* pxj
                }
            }
        }
        let invQ = gamma / Double(qX)
        for i in 0..<cols {
            out[i] = Double(accScratch[i]) * invQ
        }
    }

    // MARK: honesty

    /// Relative L2 error of `actual` against `reference`.
    public static func relativeError(_ actual: [Double], _ reference: [Double]) -> Double {
        var num = 0.0
        var den = 0.0
        for i in reference.indices {
            let d = actual[i] - reference[i]
            num += d * d
            den += reference[i] * reference[i]
        }
        if den == 0 {
            return num == 0 ? 0 : .infinity
        }
        return (num / den).squareRoot()
    }

    // MARK: GAIA fold_spectrum

    public static let gravityRest = 1.0
    public static let entropyRest = 0.62
    public static let weatherRest = 0.5
    public static let phi = 1.618033988749894848204586834365638118

    public struct Fold: Equatable, Sendable {
        public let seed: UInt32
        public let gravity: Double
        public let entropy: Double
        public let weather: Double
        public let resonance: Double
        public let dominant: String
        public let coherence: Double
    }

    public struct Sampling: Equatable, Sendable {
        public let temperature: Double
        public let topP: Double
        public let topK: UInt32
        public let seed: UInt32
    }

    /// 32-bit FNV-1a over UTF-8 prompt bytes.
    public static func seedFrom(_ prompt: String) -> UInt32 {
        var h: UInt32 = 0x811c9dc5
        for b in prompt.utf8 {
            h ^= UInt32(b)
            h = h &* 0x01000193
        }
        return h
    }

    @inline(__always)
    private static func clamp(_ v: Double, _ minV: Double, _ maxV: Double) -> Double {
        if v.isNaN || v < minV { return minV }
        if v > maxV { return maxV }
        return v
    }

    /// Fold a prompt together with passive RF spectral entropy (in bits).
    public static func foldSpectrum(
        _ prompt: String,
        rfEntropyBits: Double,
        seed: UInt32? = nil
    ) -> Fold {
        let s = seed ?? seedFrom(prompt)
        let a = s &+ 0x6d2b79f5
        var t = (a ^ (a >> 15)) &* (a | 1)
        t ^= t &+ ((t ^ (t >> 7)) &* (t | 61))
        let rand0 = Double((t ^ (t >> 14)) >> 8) / 16_777_216.0

        let rfFactor = clamp((rfEntropyBits - 2.5) * 0.05, -0.08, 0.14)
        let lenFactor = min(Double(prompt.utf8.count) / 4096.0, 0.35)
        let entropy = clamp(entropyRest + lenFactor - 0.15 + rfFactor, 0.0, 1.0)
        let weather = clamp(weatherRest + (rand0 - 0.5) * 0.4, 0.0, 1.0)
        let gravity = gravityRest
        let resonance = phi

        let names = ["gravity", "entropy", "weather", "resonance"]
        let vals = [gravity, entropy, weather, resonance]
        let rests = [gravityRest, entropyRest, weatherRest, phi]
        var dominant = "gravity"
        var best = -Double.infinity
        for i in 0..<4 {
            let d = abs(vals[i] - rests[i])
            if d > best {
                best = d
                dominant = names[i]
            }
        }

        let spread = abs(entropy - entropyRest) * 0.6 + abs(weather - weatherRest) * 0.4
        let coherence = clamp(1.0 - spread, 0.0, 1.0)

        return Fold(
            seed: s,
            gravity: gravity,
            entropy: entropy,
            weather: weather,
            resonance: resonance,
            dominant: dominant,
            coherence: coherence
        )
    }

    /// Derive sampling parameters from a Fold.
    public static func sampling(_ f: Fold) -> Sampling {
        Sampling(
            temperature: clamp(0.15 + f.entropy * 1.1, 0.05, 1.2),
            topP: clamp(0.5 + f.coherence * 0.48, 0.5, 0.98),
            topK: UInt32(clamp(20.0 + (1.0 - f.coherence) * 60.0, 20.0, 80.0) + 0.5),
            seed: f.seed
        )
    }
}
