#if canImport(XCTest)
import XCTest
@testable import GaiaMLXQuant

final class KernelTests: XCTestCase {

    func testRoundTripEveryLengthModuloFive() {
        for n in 0...32 {
            var trits = [Int8](repeating: 0, count: n)
            for i in 0..<n { trits[i] = i % 2 == 0 ? 1 : -1 }
            var packed = [UInt8](repeating: 0, count: (n + 4) / 5)
            GaiaMLXQuant.packTrits(trits, into: &packed)
            var out = [Int8](repeating: 0, count: n)
            GaiaMLXQuant.unpackTrits(packed, count: n, into: &out)
            XCTAssertEqual(out, trits, "n=\(n)")
        }
    }

    func testByte242IsAllMinusOne() {
        var out = [Int8](repeating: 0, count: 5)
        GaiaMLXQuant.unpackTrits([242], count: 5, into: &out)
        XCTAssertEqual(out, [Int8](repeating: -1, count: 5))
    }

    func testBranchlessDigitMapping() {
        XCTAssertEqual(GaiaMLXQuant.digitOf(-1), 2)
        XCTAssertEqual(GaiaMLXQuant.digitOf(0), 0)
        XCTAssertEqual(GaiaMLXQuant.digitOf(1), 1)
    }

    func testKernelMatchesDenseExactly() {
        let rows = 3, cols = 5
        let trits: [Int8] = [
            1, -1, 0, 1, -1,
            -1, 1, 1, 0, 0,
            0, 0, -1, 1, 1,
        ]
        let x: [Double] = [2, -1, 3]
        var packed = [UInt8](repeating: 0, count: (rows * cols + 4) / 5)
        GaiaMLXQuant.packTrits(trits, into: &packed)

        var ternary = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.ternaryMatMul(packed: packed, rows: rows, cols: cols, x: x, scales: nil, gamma: 1, into: &ternary)

        let dense = trits.map { Double($0) }
        var reference = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.denseMatMul(dense, rows: rows, cols: cols, x: x, into: &reference)

        XCTAssertEqual(ternary, reference)
    }

    func testPerRowScalesApply() {
        let rows = 2, cols = 5
        let trits: [Int8] = [1, -1, 0, 1, -1, 0, 1, 1, -1, 0]
        let x: [Double] = [2, 3]
        let scales: [Double] = [0.5, 2]
        var packed = [UInt8](repeating: 0, count: (rows * cols + 4) / 5)
        GaiaMLXQuant.packTrits(trits, into: &packed)
        var out = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.ternaryMatMul(packed: packed, rows: rows, cols: cols, x: x, scales: scales, gamma: 1, into: &out)

        let expected: [Double] = [
            0.5 * 1 * 2 + 2 * 0 * 3,
            0.5 * -1 * 2 + 2 * 1 * 3,
            0.5 * 0 * 2 + 2 * 1 * 3,
            0.5 * 1 * 2 + 2 * -1 * 3,
            0.5 * -1 * 2 + 2 * 0 * 3,
        ]
        XCTAssertEqual(out, expected)
    }

    func testNonAlignedRowBoundaries() {
        let rows = 4, cols = 3
        let trits: [Int8] = (0..<rows * cols).map { [-1, 0, 1][$0 % 3] }
        let x: [Double] = [1, 2, 3, 4]
        var packed = [UInt8](repeating: 0, count: (rows * cols + 4) / 5)
        GaiaMLXQuant.packTrits(trits, into: &packed)
        var out = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.ternaryMatMul(packed: packed, rows: rows, cols: cols, x: x, scales: nil, gamma: 1, into: &out)

        let dense = trits.map { Double($0) }
        var reference = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.denseMatMul(dense, rows: rows, cols: cols, x: x, into: &reference)
        XCTAssertEqual(out, reference)
    }

    func testQuantizationProducesLegalTrits() {
        let rows = 64, cols = 64
        let w: [Double] = (0..<rows * cols).map { Double($0 % 7 - 3) / 3.0 }
        let (trits, scales) = GaiaMLXQuant.quantizePerRow(w, rows: rows, cols: cols)
        for t in trits {
            XCTAssertTrue(t == -1 || t == 0 || t == 1)
        }
        XCTAssertEqual(scales.count, rows)
    }

    func testRelativeError() {
        XCTAssertEqual(GaiaMLXQuant.relativeError([1, 2, 3], [1, 2, 3]), 0)
        XCTAssertGreaterThan(GaiaMLXQuant.relativeError([2], [1]), 0.9)
    }

    func testReconstructionBoundedOnRandomWeights() {
        let rows = 64, cols = 64
        let w: [Double] = (0..<rows * cols).map { i in
            let v = Double((i * 2654435761 % 1000)) / 1000.0
            return v * 2 - 1
        }
        let x: [Double] = (0..<rows).map { Double($0 % 13) / 13.0 }
        let (trits, scales) = GaiaMLXQuant.quantizePerRow(w, rows: rows, cols: cols)
        var packed = [UInt8](repeating: 0, count: (rows * cols + 4) / 5)
        GaiaMLXQuant.packTrits(trits, into: &packed)
        var ternary = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.ternaryMatMul(packed: packed, rows: rows, cols: cols, x: x, scales: scales, gamma: 1, into: &ternary)
        var dense = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.denseMatMul(w, rows: rows, cols: cols, x: x, into: &dense)
        // The honest bound: random projections are the worst case.
        XCTAssertLessThan(GaiaMLXQuant.relativeError(ternary, dense), 0.6)
    }

    func testFoldSpectrumModulatesEntropy() {
        let prompt = "Analyze Shinjuku 2.4GHz Ch 6 saturation, BLE proximity & AirPlay RAOP surface"
        let low = GaiaMLXQuant.foldSpectrum(prompt, rfEntropyBits: 0.5)
        let high = GaiaMLXQuant.foldSpectrum(prompt, rfEntropyBits: 4.8)
        XCTAssertEqual(low.seed, high.seed)
        XCTAssertEqual(low.weather, high.weather)
        XCTAssertGreaterThan(high.entropy, low.entropy)
        let s = GaiaMLXQuant.sampling(high)
        XCTAssertTrue((0.05...1.2).contains(s.temperature))
        XCTAssertTrue((20...80).contains(s.topK))
    }

    func testTernaryMatMulPqMatchesFloatTo1e5() {
        let rows = 32, cols = 32
        let w: [Double] = (0..<rows * cols).map { Double(($0 * 17 + 3) % 23) / 11.0 - 1.0 }
        let x: [Double] = (0..<rows).map { Double(($0 * 11 + 5) % 19) / 9.0 - 1.0 }
        let (trits, scales) = GaiaMLXQuant.quantizePerRow(w, rows: rows, cols: cols)
        var packed = [UInt8](repeating: 0, count: (rows * cols + 4) / 5)
        GaiaMLXQuant.packTrits(trits, into: &packed)

        var outFloat = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.ternaryMatMul(packed: packed, rows: rows, cols: cols, x: x, scales: scales, gamma: 1.0, into: &outFloat)

        var pX = [Int32](repeating: 0, count: rows)
        GaiaMLXQuant.quantizePqInto(x, q: GaiaMLXQuant.pqDefaultQ, into: &pX)
        var acc = [Int64](repeating: 0, count: cols)
        var outPq = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.ternaryMatMulPq(
            packed: packed,
            rows: rows,
            cols: cols,
            pX: pX,
            qX: GaiaMLXQuant.pqDefaultQ,
            scales: scales,
            gamma: 1.0,
            accScratch: &acc,
            into: &outPq
        )
        XCTAssertLessThan(GaiaMLXQuant.relativeError(outPq, outFloat), 1e-5)
    }
}
#elseif canImport(Testing)
import Testing
@testable import GaiaMLXQuant

struct KernelTests {

    @Test func roundTripEveryLengthModuloFive() {
        for n in 0...32 {
            var trits = [Int8](repeating: 0, count: n)
            for i in 0..<n { trits[i] = i % 2 == 0 ? 1 : -1 }
            var packed = [UInt8](repeating: 0, count: (n + 4) / 5)
            GaiaMLXQuant.packTrits(trits, into: &packed)
            var out = [Int8](repeating: 0, count: n)
            GaiaMLXQuant.unpackTrits(packed, count: n, into: &out)
            #expect(out == trits)
        }
    }

    @Test func byte242IsAllMinusOne() {
        var out = [Int8](repeating: 0, count: 5)
        GaiaMLXQuant.unpackTrits([242], count: 5, into: &out)
        #expect(out == [Int8](repeating: -1, count: 5))
    }

    @Test func branchlessDigitMapping() {
        #expect(GaiaMLXQuant.digitOf(-1) == 2)
        #expect(GaiaMLXQuant.digitOf(0) == 0)
        #expect(GaiaMLXQuant.digitOf(1) == 1)
    }

    @Test func kernelMatchesDenseExactly() {
        let rows = 3, cols = 5
        let trits: [Int8] = [
            1, -1, 0, 1, -1,
            -1, 1, 1, 0, 0,
            0, 0, -1, 1, 1,
        ]
        let x: [Double] = [2, -1, 3]
        var packed = [UInt8](repeating: 0, count: (rows * cols + 4) / 5)
        GaiaMLXQuant.packTrits(trits, into: &packed)

        var ternary = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.ternaryMatMul(packed: packed, rows: rows, cols: cols, x: x, scales: nil, gamma: 1, into: &ternary)

        let dense = trits.map { Double($0) }
        var reference = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.denseMatMul(dense, rows: rows, cols: cols, x: x, into: &reference)

        #expect(ternary == reference)
    }

    @Test func perRowScalesApply() {
        let rows = 2, cols = 5
        let trits: [Int8] = [1, -1, 0, 1, -1, 0, 1, 1, -1, 0]
        let x: [Double] = [2, 3]
        let scales: [Double] = [0.5, 2]
        var packed = [UInt8](repeating: 0, count: (rows * cols + 4) / 5)
        GaiaMLXQuant.packTrits(trits, into: &packed)
        var out = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.ternaryMatMul(packed: packed, rows: rows, cols: cols, x: x, scales: scales, gamma: 1, into: &out)

        let expected: [Double] = [
            0.5 * 1 * 2 + 2 * 0 * 3,
            0.5 * -1 * 2 + 2 * 1 * 3,
            0.5 * 0 * 2 + 2 * 1 * 3,
            0.5 * 1 * 2 + 2 * -1 * 3,
            0.5 * -1 * 2 + 2 * 0 * 3,
        ]
        #expect(out == expected)
    }

    @Test func nonAlignedRowBoundaries() {
        let rows = 4, cols = 3
        let trits: [Int8] = (0..<rows * cols).map { [-1, 0, 1][$0 % 3] }
        let x: [Double] = [1, 2, 3, 4]
        var packed = [UInt8](repeating: 0, count: (rows * cols + 4) / 5)
        GaiaMLXQuant.packTrits(trits, into: &packed)
        var out = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.ternaryMatMul(packed: packed, rows: rows, cols: cols, x: x, scales: nil, gamma: 1, into: &out)

        let dense = trits.map { Double($0) }
        var reference = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.denseMatMul(dense, rows: rows, cols: cols, x: x, into: &reference)
        #expect(out == reference)
    }

    @Test func quantizationProducesLegalTrits() {
        let rows = 64, cols = 64
        let w: [Double] = (0..<rows * cols).map { Double($0 % 7 - 3) / 3.0 }
        let (trits, scales) = GaiaMLXQuant.quantizePerRow(w, rows: rows, cols: cols)
        for t in trits {
            #expect(t == -1 || t == 0 || t == 1)
        }
        #expect(scales.count == rows)
    }

    @Test func relativeError() {
        #expect(GaiaMLXQuant.relativeError([1, 2, 3], [1, 2, 3]) == 0)
        #expect(GaiaMLXQuant.relativeError([2], [1]) > 0.9)
    }

    @Test func reconstructionBoundedOnRandomWeights() {
        let rows = 64, cols = 64
        let w: [Double] = (0..<rows * cols).map { i in
            let v = Double((i * 2654435761 % 1000)) / 1000.0
            return v * 2 - 1
        }
        let x: [Double] = (0..<rows).map { Double($0 % 13) / 13.0 }
        let (trits, scales) = GaiaMLXQuant.quantizePerRow(w, rows: rows, cols: cols)
        var packed = [UInt8](repeating: 0, count: (rows * cols + 4) / 5)
        GaiaMLXQuant.packTrits(trits, into: &packed)
        var ternary = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.ternaryMatMul(packed: packed, rows: rows, cols: cols, x: x, scales: scales, gamma: 1, into: &ternary)
        var dense = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.denseMatMul(w, rows: rows, cols: cols, x: x, into: &dense)
        #expect(GaiaMLXQuant.relativeError(ternary, dense) < 0.6)
    }

    @Test func foldSpectrumModulatesEntropy() {
        let prompt = "Analyze Shinjuku 2.4GHz Ch 6 saturation, BLE proximity & AirPlay RAOP surface"
        let low = GaiaMLXQuant.foldSpectrum(prompt, rfEntropyBits: 0.5)
        let high = GaiaMLXQuant.foldSpectrum(prompt, rfEntropyBits: 4.8)
        #expect(low.seed == high.seed)
        #expect(low.weather == high.weather)
        #expect(high.entropy > low.entropy)
        let s = GaiaMLXQuant.sampling(high)
        #expect((0.05...1.2).contains(s.temperature))
        #expect((20...80).contains(s.topK))
    }

    @Test func ternaryMatMulPqMatchesFloatTo1e5() {
        let rows = 32, cols = 32
        let w: [Double] = (0..<rows * cols).map { Double(($0 * 17 + 3) % 23) / 11.0 - 1.0 }
        let x: [Double] = (0..<rows).map { Double(($0 * 11 + 5) % 19) / 9.0 - 1.0 }
        let (trits, scales) = GaiaMLXQuant.quantizePerRow(w, rows: rows, cols: cols)
        var packed = [UInt8](repeating: 0, count: (rows * cols + 4) / 5)
        GaiaMLXQuant.packTrits(trits, into: &packed)

        var outFloat = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.ternaryMatMul(packed: packed, rows: rows, cols: cols, x: x, scales: scales, gamma: 1.0, into: &outFloat)

        var pX = [Int32](repeating: 0, count: rows)
        GaiaMLXQuant.quantizePqInto(x, q: GaiaMLXQuant.pqDefaultQ, into: &pX)
        var acc = [Int64](repeating: 0, count: cols)
        var outPq = [Double](repeating: 0, count: cols)
        GaiaMLXQuant.ternaryMatMulPq(
            packed: packed,
            rows: rows,
            cols: cols,
            pX: pX,
            qX: GaiaMLXQuant.pqDefaultQ,
            scales: scales,
            gamma: 1.0,
            accScratch: &acc,
            into: &outPq
        )
        #expect(GaiaMLXQuant.relativeError(outPq, outFloat) < 1e-5)
    }
}
#endif
