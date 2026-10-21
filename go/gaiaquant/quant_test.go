package gaiaquant

import "testing"

func TestRoundTripEveryLength(t *testing.T) {
	for n := 0; n <= 32; n++ {
		trits := make([]int8, n)
		for i := range trits {
			if i%2 == 0 {
				trits[i] = 1
			} else {
				trits[i] = -1
			}
		}
		packed := make([]uint8, (n+TritPerByte-1)/TritPerByte)
		PackTrits(trits, packed)
		out := make([]int8, n)
		UnpackTrits(packed, n, out)
		for i := range trits {
			if out[i] != trits[i] {
				t.Fatalf("n=%d i=%d: got %d want %d", n, i, out[i], trits[i])
			}
		}
	}
}

func TestByte242IsAllMinusOne(t *testing.T) {
	packed := []uint8{242}
	out := make([]int8, 5)
	UnpackTrits(packed, 5, out)
	for i, v := range out {
		if v != -1 {
			t.Fatalf("slot %d: got %d want -1", i, v)
		}
	}
}

func TestKernelMatchesDenseExactly(t *testing.T) {
	rows, cols := 3, 5
	trits := []int8{
		1, -1, 0, 1, -1,
		-1, 1, 1, 0, 0,
		0, 0, -1, 1, 1,
	}
	x := []float64{2, -1, 3}
	packed := make([]uint8, (rows*cols+TritPerByte-1)/TritPerByte)
	PackTrits(trits, packed)

	ternary := make([]float64, cols)
	TernaryMatMul(packed, rows, cols, x, nil, 1, ternary)

	dense := make([]float64, rows*cols)
	for i, v := range trits {
		dense[i] = float64(v)
	}
	reference := make([]float64, cols)
	DenseMatMul(dense, rows, cols, x, reference)

	for i := range reference {
		if ternary[i] != reference[i] {
			t.Fatalf("i=%d: got %v want %v", i, ternary[i], reference[i])
		}
	}
}

func TestPerRowScalesApply(t *testing.T) {
	rows, cols := 2, 5
	trits := []int8{1, -1, 0, 1, -1, 0, 1, 1, -1, 0}
	x := []float64{2, 3}
	scales := []float64{0.5, 2}
	packed := make([]uint8, (rows*cols+TritPerByte-1)/TritPerByte)
	PackTrits(trits, packed)
	out := make([]float64, cols)
	TernaryMatMul(packed, rows, cols, x, scales, 1, out)

	expected := []float64{
		0.5*1*2 + 2*0*3,
		0.5*-1*2 + 2*1*3,
		0.5*0*2 + 2*1*3,
		0.5*1*2 + 2*-1*3,
		0.5*-1*2 + 2*0*3,
	}
	for i := range out {
		if out[i] != expected[i] {
			t.Fatalf("i=%d: got %v want %v", i, out[i], expected[i])
		}
	}
}

func TestNonAlignedRowBoundaries(t *testing.T) {
	// cols=3: rows cross byte boundaries (3 % 5 != 0).
	rows, cols := 4, 3
	trits := make([]int8, rows*cols)
	for i := range trits {
		trits[i] = []int8{-1, 0, 1}[i%3]
	}
	x := []float64{1, 2, 3, 4}
	packed := make([]uint8, (rows*cols+TritPerByte-1)/TritPerByte)
	PackTrits(trits, packed)

	out := make([]float64, cols)
	TernaryMatMul(packed, rows, cols, x, nil, 1, out)

	dense := make([]float64, rows*cols)
	for i, v := range trits {
		dense[i] = float64(v)
	}
	reference := make([]float64, cols)
	DenseMatMul(dense, rows, cols, x, reference)

	for i := range reference {
		if out[i] != reference[i] {
			t.Fatalf("i=%d: got %v want %v", i, out[i], reference[i])
		}
	}
}

func TestZeroRowsAreSkipped(t *testing.T) {
	rows, cols := 3, 5
	trits := make([]int8, 15)
	for i := range trits {
		trits[i] = 1
	}
	x := []float64{0, 2, 0}
	packed := make([]uint8, (rows*cols+TritPerByte-1)/TritPerByte)
	PackTrits(trits, packed)
	out := make([]float64, cols)
	TernaryMatMul(packed, rows, cols, x, nil, 1, out)
	for i, v := range out {
		if v != 2 {
			t.Fatalf("i=%d: got %v want 2", i, v)
		}
	}
}

func TestRelativeError(t *testing.T) {
	if RelativeError([]float64{1, 2, 3}, []float64{1, 2, 3}) != 0 {
		t.Fatal("identical vectors must have zero error")
	}
	if RelativeError([]float64{2}, []float64{1}) <= 0.9 {
		t.Fatal("doubled vector must have error > 0.9")
	}
}

func TestQuantizationProducesLegalTrits(t *testing.T) {
	rows, cols := 64, 64
	w := make([]float64, rows*cols)
	for i := range w {
		w[i] = float64(i%7-3) / 3.0
	}
	trits, scales := QuantizePerRow(w, rows, cols)
	for i, trit := range trits {
		if trit != -1 && trit != 0 && trit != 1 {
			t.Fatalf("i=%d: illegal trit %d", i, trit)
		}
	}
	if len(scales) != rows {
		t.Fatalf("scales len %d, want %d", len(scales), rows)
	}
}

func TestFoldSpectrumModulatesEntropy(t *testing.T) {
	prompt := "Analyze Shinjuku 2.4GHz Ch 6 saturation, BLE proximity & AirPlay RAOP surface"
	low := FoldSpectrum(prompt, 0.5, nil)
	high := FoldSpectrum(prompt, 4.8, nil)
	if low.Seed != high.Seed {
		t.Fatalf("seed mismatch: %d vs %d", low.Seed, high.Seed)
	}
	if low.Weather != high.Weather {
		t.Fatalf("weather mismatch: %v vs %v", low.Weather, high.Weather)
	}
	if !(high.Entropy > low.Entropy) {
		t.Fatalf("expected high.Entropy (%v) > low.Entropy (%v)", high.Entropy, low.Entropy)
	}
	s := DeriveSampling(high)
	if s.Temperature < 0.05 || s.Temperature > 1.2 || s.TopK < 20 || s.TopK > 80 {
		t.Fatalf("sampling out of bounds: %+v", s)
	}

	lowPq := FoldSpectrumPq(prompt, 524288, nil)
	highPq := FoldSpectrumPq(prompt, 5033165, nil)
	if lowPq.Seed != low.Seed || !(highPq.EntropyP > lowPq.EntropyP) {
		t.Fatalf("FoldSpectrumPq mismatch: low=%+v high=%+v", lowPq, highPq)
	}
}

func TestTernaryMatMulPqMatchesExactInteger(t *testing.T) {
	rows, cols := 4, 7
	trits := make([]int8, rows*cols)
	for i := range trits {
		trits[i] = []int8{-1, 0, 1}[i%3]
	}
	pX := []int32{1048576, -524288, 2097152, 262144}
	packed := make([]uint8, (rows*cols+TritPerByte-1)/TritPerByte)
	PackTrits(trits, packed)
	pOut := make([]int64, cols)
	TernaryMatMulPq(packed, rows, cols, pX, nil, int32(PqDefaultQ), pOut)

	for c := 0; c < cols; c++ {
		var want int64
		for r := 0; r < rows; r++ {
			want += int64(trits[r*cols+c]) * int64(pX[r])
		}
		if pOut[c] != want {
			t.Fatalf("col %d: got %d want %d", c, pOut[c], want)
		}
	}

	// Verify 64-trit Quantum Bitplane POPCNT kernel matches exact integer reference
	qRows, qCols := 68, 9
	words := (qRows + 63) / 64
	qTrits := make([]int8, qRows*qCols)
	for i := range qTrits {
		qTrits[i] = []int8{-1, 0, 1}[(i*5+1)%3]
	}
	qPx := make([]int32, qRows)
	for r := 0; r < qRows; r++ {
		qPx[r] = []int32{-int32(PqDefaultQ), 0, int32(PqDefaultQ)}[(r*3+2)%3]
	}
	wPos := make([]uint64, qCols*words)
	wNeg := make([]uint64, qCols*words)
	xPos := make([]uint64, words)
	xNeg := make([]uint64, words)
	PackQuantumBitplanesIMajor(qTrits, qRows, qCols, wPos, wNeg)
	scaleQx := QuantizeQuantumActPq(qPx, xPos, xNeg)
	qOut := make([]int64, qCols)
	QuantumBitplaneMatMulPq(wPos, wNeg, qRows, qCols, xPos, xNeg, scaleQx, qOut)
	for c := 0; c < qCols; c++ {
		var tritDot int64
		for r := 0; r < qRows; r++ {
			var xt int64
			if qPx[r] > 0 {
				xt = 1
			} else if qPx[r] < 0 {
				xt = -1
			}
			tritDot += int64(qTrits[r*qCols+c]) * xt
		}
		want := tritDot * int64(scaleQx)
		if qOut[c] != want {
			t.Fatalf("quantum bitplane col %d: got %d want %d", c, qOut[c], want)
		}
	}
}
