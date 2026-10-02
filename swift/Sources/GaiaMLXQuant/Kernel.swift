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
            for c in 0..<cols {
                let q = (w[r * cols + c] / scale).rounded()
                trits[r * cols + c] = q > 0 ? 1 : (q < 0 ? -1 : 0)
            }
        }
        return (trits, scales)
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
}
