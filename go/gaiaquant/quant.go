// Package gaiaquant is the Go lane of the GAIA-MLX-QUANT substrate —
// the same ternary b1.58 math as the Rust core and the C99 lane, in pure,
// allocation-lean Go. "Fine tuned for life": the hot loops touch only
// slices and static LUTs, allocate nothing, and read like a sentence.
package gaiaquant

import (
	"math"
	"math/bits"
)

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

// GAIA field rest constants.
const (
	GravityRest = 1.0
	EntropyRest = 0.62
	WeatherRest = 0.5
	Phi         = 1.618033988749894848204586834365638118
)

// Fold holds the 4-layer GAIA fold result.
type Fold struct {
	Seed      uint32
	Gravity   float64
	Entropy   float64
	Weather   float64
	Resonance float64
	Dominant  string
	Coherence float64
}

// Sampling holds deterministic sampling parameters derived from a Fold.
type Sampling struct {
	Temperature float64
	TopP        float64
	TopK        uint32
	Seed        uint32
}

// SeedFrom computes 32-bit FNV-1a over UTF-8 prompt bytes.
func SeedFrom(prompt string) uint32 {
	var h uint32 = 0x811c9dc5
	for i := 0; i < len(prompt); i++ {
		h ^= uint32(prompt[i])
		h *= 0x01000193
	}
	return h
}

func clamp64(v, minV, maxV float64) float64 {
	if math.IsNaN(v) || v < minV {
		return minV
	}
	if v > maxV {
		return maxV
	}
	return v
}

// FoldSpectrum folds a prompt with passive RF Shannon entropy (in bits),
// matching gaia_mlx_quant::fold_spectrum across the Rust, Swift, C99, and JS lanes.
func FoldSpectrum(prompt string, rfEntropyBits float64, explicitSeed *uint32) Fold {
	s := SeedFrom(prompt)
	if explicitSeed != nil {
		s = *explicitSeed
	}
	a := s + 0x6d2b79f5
	t := (a ^ (a >> 15)) * (a | 1)
	t ^= t + ((t ^ (t >> 7)) * (t | 61))
	rand0 := float64((t^(t>>14))>>8) / 16777216.0

	rfFactor := clamp64((rfEntropyBits-2.5)*0.05, -0.08, 0.14)
	lenFactor := math.Min(float64(len(prompt))/4096.0, 0.35)
	entropy := clamp64(EntropyRest+lenFactor-0.15+rfFactor, 0.0, 1.0)
	weather := clamp64(WeatherRest+(rand0-0.5)*0.4, 0.0, 1.0)
	gravity := GravityRest
	resonance := Phi

	dominant := "gravity"
	best := math.Inf(-1)
	names := [4]string{"gravity", "entropy", "weather", "resonance"}
	vals := [4]float64{gravity, entropy, weather, resonance}
	rests := [4]float64{GravityRest, EntropyRest, WeatherRest, Phi}
	for i := 0; i < 4; i++ {
		d := math.Abs(vals[i] - rests[i])
		if d > best {
			best = d
			dominant = names[i]
		}
	}

	spread := math.Abs(entropy-EntropyRest)*0.6 + math.Abs(weather-WeatherRest)*0.4
	coherence := clamp64(1.0-spread, 0.0, 1.0)

	return Fold{
		Seed:      s,
		Gravity:   gravity,
		Entropy:   entropy,
		Weather:   weather,
		Resonance: resonance,
		Dominant:  dominant,
		Coherence: coherence,
	}
}

// DeriveSampling derives sampling parameters from a Fold.
func DeriveSampling(f Fold) Sampling {
	return Sampling{
		Temperature: clamp64(0.15+f.Entropy*1.1, 0.05, 1.2),
		TopP:        clamp64(0.5+f.Coherence*0.48, 0.5, 0.98),
		TopK:        uint32(clamp64(20.0+(1.0-f.Coherence)*60.0, 20.0, 80.0) + 0.5),
		Seed:        f.Seed,
	}
}

// ── Pure-Integer p/q ("Counting Divided" · Zero Floating Point) Go Lane ────

// PqDefaultQ is the canonical rational denominator q = 2^20 = 1,048,576.
const PqDefaultQ int64 = 1048576

// FoldPq holds the 4-layer GAIA fold result in pure integer p/q (q = 2^20), zero floats.
type FoldPq struct {
	Seed       uint32
	Q          int64
	GravityP   int64
	EntropyP   int64
	WeatherP   int64
	ResonanceP int64
	Dominant   string
	CoherenceP int64
}

func clampI64(v, minV, maxV int64) int64 {
	if v < minV {
		return minV
	}
	if v > maxV {
		return maxV
	}
	return v
}

func absI64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

// FoldSpectrumPq folds a prompt with passive RF Shannon entropy numerator rfEntropyP
// (in q = 2^20) using 100% integer arithmetic and zero floating-point operations.
func FoldSpectrumPq(prompt string, rfEntropyP int64, explicitSeed *uint32) FoldPq {
	s := SeedFrom(prompt)
	if explicitSeed != nil {
		s = *explicitSeed
	}
	a := s + 0x6d2b79f5
	t := (a ^ (a >> 15)) * (a | 1)
	t ^= t + ((t ^ (t >> 7)) * (t | 61))
	randP := int64((t ^ (t >> 14)) >> 12)

	rfFactorP := clampI64((rfEntropyP-2621440)/20, -83886, 146801)
	lenP := int64(len(prompt)) << 8
	if lenP > 367002 {
		lenP = 367002
	}
	entropyP := clampI64(650117+lenP-157286+rfFactorP, 0, PqDefaultQ)
	weatherP := clampI64(524288+((randP-524288)*2)/5, 0, PqDefaultQ)
	gravityP := PqDefaultQ
	resonanceP := int64(1696632)

	names := [4]string{"gravity", "entropy", "weather", "resonance"}
	vals := [4]int64{gravityP, entropyP, weatherP, resonanceP}
	rests := [4]int64{1048576, 650117, 524288, 1696632}
	dominant := "gravity"
	var best int64 = -1
	for i := 0; i < 4; i++ {
		d := absI64(vals[i] - rests[i])
		if d > best {
			best = d
			dominant = names[i]
		}
	}

	spreadP := (absI64(entropyP-650117)*3 + absI64(weatherP-524288)*2) / 5
	coherenceP := clampI64(PqDefaultQ-spreadP, 0, PqDefaultQ)

	return FoldPq{
		Seed:       s,
		Q:          PqDefaultQ,
		GravityP:   gravityP,
		EntropyP:   entropyP,
		WeatherP:   weatherP,
		ResonanceP: resonanceP,
		Dominant:   dominant,
		CoherenceP: coherenceP,
	}
}

// QuantizePerRowPq quantizes integer numerators wP ([rows, cols]) to trits {-1, 0, +1}
// and per-row absmean scale numerators scalesP using exact integer inequality
// |v| * 2 * cols >= sumAbs (zero floats, zero divisions per weight).
func QuantizePerRowPq(wP []int32, rows, cols int, q int32) (trits []int8, scalesP []int32) {
	trits = make([]int8, len(wP))
	scalesP = make([]int32, rows)
	cols64 := int64(cols)
	twoCols := cols64 * 2
	for r := 0; r < rows; r++ {
		row := wP[r*cols : (r+1)*cols]
		var sumAbs int64
		for _, v := range row {
			sumAbs += absI64(int64(v))
		}
		if sumAbs == 0 {
			scalesP[r] = q
			continue
		}
		scalesP[r] = int32((sumAbs + (cols64 >> 1)) / cols64)
		for c, v := range row {
			v64 := int64(v)
			if absI64(v64)*twoCols >= sumAbs {
				if v64 > 0 {
					trits[r*cols+c] = 1
				} else {
					trits[r*cols+c] = -1
				}
			}
		}
	}
	return trits, scalesP
}

// TernaryMatMulPq is the pure-integer j-major 5-trits/byte kernel in p/q (zero floats).
func TernaryMatMulPq(packed []uint8, rows, cols int, pX, pScales []int32, qScale int32, pOut []int64) {
	for i := range pOut[:cols] {
		pOut[i] = 0
	}
	qS := int64(qScale)
	if qS <= 0 {
		qS = 1
	}
	halfQ := qS >> 1
	for j := 0; j < rows; j++ {
		xj := int64(pX[j])
		if xj == 0 {
			continue
		}
		if pScales != nil {
			prod := xj * int64(pScales[j])
			if prod >= 0 {
				xj = (prod + halfQ) / qS
			} else {
				xj = (prod - halfQ) / qS
			}
			if xj == 0 {
				continue
			}
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
			if base >= startT && base+TritPerByte <= endT {
				col := base - startT
				pOut[col+0] += int64(T0[byte_]) * xj
				pOut[col+1] += int64(T1[byte_]) * xj
				pOut[col+2] += int64(T2[byte_]) * xj
				pOut[col+3] += int64(T3[byte_]) * xj
				pOut[col+4] += int64(T4[byte_]) * xj
			} else {
				if t := base; t >= startT && t < endT {
					pOut[t-startT] += int64(T0[byte_]) * xj
				}
				if t := base + 1; t >= startT && t < endT {
					pOut[t-startT] += int64(T1[byte_]) * xj
				}
				if t := base + 2; t >= startT && t < endT {
					pOut[t-startT] += int64(T2[byte_]) * xj
				}
				if t := base + 3; t >= startT && t < endT {
					pOut[t-startT] += int64(T3[byte_]) * xj
				}
				if t := base + 4; t >= startT && t < endT {
					pOut[t-startT] += int64(T4[byte_]) * xj
				}
			}
		}
	}
}

// PackQuantumBitplanesIMajor packs j-major trits ([rows, cols]) into column-major
// 64-trit quantum bitplane words (wPosU64, wNegU64), each of length cols * ceil(rows/64).
func PackQuantumBitplanesIMajor(trits []int8, rows, cols int, wPosU64, wNegU64 []uint64) {
	words := (rows + 63) / 64
	for i := 0; i < cols; i++ {
		for w := 0; w < words; w++ {
			var pMask, nMask uint64
			jBase := w * 64
			jEnd := jBase + 64
			if jEnd > rows {
				jEnd = rows
			}
			for b := 0; b < jEnd-jBase; b++ {
				t := trits[(jBase+b)*cols+i]
				if t > 0 {
					pMask |= 1 << uint(b)
				} else if t < 0 {
					nMask |= 1 << uint(b)
				}
			}
			wPosU64[i*words+w] = pMask
			wNegU64[i*words+w] = nMask
		}
	}
}

// QuantizeQuantumActPq quantizes integer activation numerators pX into 64-trit
// quantum bitplanes (xPosU64, xNegU64) and returns the absmean integer scale pScaleX (zero floats).
func QuantizeQuantumActPq(pX []int32, xPosU64, xNegU64 []uint64) int32 {
	n := len(pX)
	words := (n + 63) / 64
	for w := 0; w < words; w++ {
		xPosU64[w] = 0
		xNegU64[w] = 0
	}
	if n == 0 {
		return 0
	}
	var sumAbs int64
	for _, v := range pX {
		sumAbs += absI64(int64(v))
	}
	if sumAbs == 0 {
		return 0
	}
	n64 := int64(n)
	twoN := n64 * 2
	for w := 0; w < words; w++ {
		var pm, nm uint64
		jBase := w * 64
		jEnd := jBase + 64
		if jEnd > n {
			jEnd = n
		}
		for b := 0; b < jEnd-jBase; b++ {
			v := int64(pX[jBase+b])
			if absI64(v)*twoN >= sumAbs {
				if v > 0 {
					pm |= 1 << uint(b)
				} else if v < 0 {
					nm |= 1 << uint(b)
				}
			}
		}
		xPosU64[w] = pm
		xNegU64[w] = nm
	}
	return int32((sumAbs + (n64 >> 1)) / n64)
}

// QuantumBitplaneMatMulPq contracts 64-trit quantum bitplanes (wPosU64, wNegU64)
// against (xPosU64, xNegU64) using hardware bits.OnesCount64 (POPCNT / NEON CNT)
// with 4-column register blocking and zero floating-point operations.
func QuantumBitplaneMatMulPq(wPosU64, wNegU64 []uint64, rows, cols int, xPosU64, xNegU64 []uint64, pScaleX int32, pOut []int64) {
	words := (rows + 63) / 64
	sx := int64(pScaleX)
	i := 0
	for i+4 <= cols {
		wp0, wn0 := wPosU64[(i+0)*words:(i+1)*words], wNegU64[(i+0)*words:(i+1)*words]
		wp1, wn1 := wPosU64[(i+1)*words:(i+2)*words], wNegU64[(i+1)*words:(i+2)*words]
		wp2, wn2 := wPosU64[(i+2)*words:(i+3)*words], wNegU64[(i+2)*words:(i+3)*words]
		wp3, wn3 := wPosU64[(i+3)*words:(i+4)*words], wNegU64[(i+3)*words:(i+4)*words]
		var a0, a1, a2, a3 int64
		for w := 0; w < words; w++ {
			xp, xn := xPosU64[w], xNegU64[w]
			a0 += int64(bits.OnesCount64((wp0[w]&xp)|(wn0[w]&xn)) - bits.OnesCount64((wp0[w]&xn)|(wn0[w]&xp)))
			a1 += int64(bits.OnesCount64((wp1[w]&xp)|(wn1[w]&xn)) - bits.OnesCount64((wp1[w]&xn)|(wn1[w]&xp)))
			a2 += int64(bits.OnesCount64((wp2[w]&xp)|(wn2[w]&xn)) - bits.OnesCount64((wp2[w]&xn)|(wn2[w]&xp)))
			a3 += int64(bits.OnesCount64((wp3[w]&xp)|(wn3[w]&xn)) - bits.OnesCount64((wp3[w]&xn)|(wn3[w]&xp)))
		}
		pOut[i+0] = a0 * sx
		pOut[i+1] = a1 * sx
		pOut[i+2] = a2 * sx
		pOut[i+3] = a3 * sx
		i += 4
	}
	for ; i < cols; i++ {
		wp, wn := wPosU64[i*words:(i+1)*words], wNegU64[i*words:(i+1)*words]
		var a0 int64
		for w := 0; w < words; w++ {
			xp, xn := xPosU64[w], xNegU64[w]
			a0 += int64(bits.OnesCount64((wp[w]&xp)|(wn[w]&xn)) - bits.OnesCount64((wp[w]&xn)|(wn[w]&xp)))
		}
		pOut[i] = a0 * sx
	}
}
