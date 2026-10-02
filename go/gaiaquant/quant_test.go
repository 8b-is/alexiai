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
