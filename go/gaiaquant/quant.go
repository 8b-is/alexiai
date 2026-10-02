// Package gaiaquant is the Go lane of the GAIA-MLX-QUANT substrate —
// the same ternary b1.58 math as the Rust core and the C99 lane, in pure,
// allocation-lean Go. "Fine tuned for life": the hot loops touch only
// slices and static LUTs, allocate nothing, and read like a sentence.
package gaiaquant

import "math"

// TritPerByte is the packing density: five base-3 digits per byte.
const TritPerByte = 5

// Pow3 holds the base-3 place values.
var Pow3 = [TritPerByte]uint8{1, 3, 9, 27, 81}

// The five slot LUTs: T0[b] is the trit in slot 0 of byte b. Built once at
// init, zero division and zero modulo in the hot path.
var T0, T1, T2, T3, T4 [256]int8

func init() {
	for b := 0; b < 256; b++ {
		T0[b] = slotTrit(b, 0)
		T1[b] = slotTrit(b, 1)
		T2[b] = slotTrit(b, 2)
		T3[b] = slotTrit(b, 3)
		T4[b] = slotTrit(b, 4)
	}
}

func slotTrit(b, slot int) int8 {
	digit := (b / int(Pow3[slot])) % 3
	if digit == 2 {
		return -1
	}
	return int8(digit)
}

// digitOf maps {-1, 0, +1} to {2, 0, 1} with no branch:
// for t = -1, uint8(t) = 0xFF, and 0xFF + (0xFF>>7)*3 wraps to 2.
func digitOf(t int8) uint8 {
	return uint8(t) + (uint8(t)>>7)*3
}

// PackTrits packs trits into a caller-owned byte buffer (must be zeroed;
// digits accumulate with +=). Unused trailing slots stay zero.
func PackTrits(trits []int8, out []uint8) {
	for i, t := range trits {
		out[i/TritPerByte] += Pow3[i%TritPerByte] * digitOf(t)
	}
}

// UnpackTrits recovers trits from a packed buffer.
func UnpackTrits(bytes []uint8, count int, out []int8) {
	for i, b := range bytes {
		base := i * TritPerByte
		if base >= count {
			break
		}
		out[base] = T0[b]
		if base+1 < count {
			out[base+1] = T1[b]
		}
		if base+2 < count {
			out[base+2] = T2[b]
		}
		if base+3 < count {
			out[base+3] = T3[b]
		}
		if base+4 < count {
			out[base+4] = T4[b]
		}
	}
}

// TernaryMatMul is the masked kernel, j-major (the JS-parity layout):
//
//	Y[i] = gamma * sum_j ( W+[j,i] - W-[j,i] ) * scale[j] * x[j]
//
// Accumulation is float64 so long columns never drift. The hot path
// allocates nothing.
func TernaryMatMul(packed []uint8, rows, cols int, x, scales []float64, gamma float64, out []float64) {
	for i := range out[:cols] {
		out[i] = 0
	}
	for j := 0; j < rows; j++ {
		xv := x[j]
		if xv == 0 {
			continue
		}
		xj := xv * gamma
		if scales != nil {
			xj *= scales[j]
		}
		startT := j * cols
		endT := startT + cols
		startB := startT / TritPerByte
		endB := (endT + TritPerByte - 1) / TritPerByte
		for b := startB; b < endB; b++ {
			byte_ := packed[b]
			if byte_ == 0 {
				continue
			}
			base := b * TritPerByte
			if t := base; t >= startT && t < endT {
				out[t-startT] += float64(T0[byte_]) * xj
			}
			if t := base + 1; t >= startT && t < endT {
				out[t-startT] += float64(T1[byte_]) * xj
			}
			if t := base + 2; t >= startT && t < endT {
				out[t-startT] += float64(T2[byte_]) * xj
			}
			if t := base + 3; t >= startT && t < endT {
				out[t-startT] += float64(T3[byte_]) * xj
			}
			if t := base + 4; t >= startT && t < endT {
				out[t-startT] += float64(T4[byte_]) * xj
			}
		}
	}
}

// DenseMatMul is the float64 reference.
func DenseMatMul(w []float64, rows, cols int, x []float64, out []float64) {
	for i := range out[:cols] {
		out[i] = 0
	}
	for j := 0; j < rows; j++ {
		xv := x[j]
		if xv == 0 {
			continue
		}
		row := w[j*cols : (j+1)*cols]
		for i, wv := range row {
			out[i] += wv * xv
		}
	}
}

// QuantizePerRow quantizes row-major weights to {-1, 0, +1} with per-row
// absmean scales — the scheme real b1.58 models are trained into.
func QuantizePerRow(w []float64, rows, cols int) (trits []int8, scales []float64) {
	trits = make([]int8, len(w))
	scales = make([]float64, rows)
	for r := 0; r < rows; r++ {
		row := w[r*cols : (r+1)*cols]
		var sum float64
		for _, v := range row {
			sum += math.Abs(v)
		}
		scale := sum / float64(cols)
		if scale == 0 {
			scale = 1
		}
		scales[r] = scale
		for c, v := range row {
			q := math.Round(v / scale)
			switch {
			case q > 0:
				trits[r*cols+c] = 1
			case q < 0:
				trits[r*cols+c] = -1
			}
		}
	}
	return trits, scales
}

// RelativeError is the relative L2 error of actual against reference.
func RelativeError(actual, reference []float64) float64 {
	var num, den float64
	for i := range reference {
		d := actual[i] - reference[i]
		num += d * d
		den += reference[i] * reference[i]
	}
	if den == 0 {
		if num == 0 {
			return 0
		}
		return math.Inf(1)
	}
	return math.Sqrt(num / den)
}

// BitsPerWeight is log2(3), the packing's exact budget.
const BitsPerWeight = 1.584962500721156

// CompressionVsF64 is the density win over float64.
const CompressionVsF64 = 64.0 / BitsPerWeight
