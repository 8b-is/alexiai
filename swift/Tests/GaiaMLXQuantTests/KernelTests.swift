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
}
