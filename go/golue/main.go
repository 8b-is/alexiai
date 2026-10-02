// golue — the Go glue.
//
// The supervisor between the lanes: it makes sure the Rust app binary exists
// (building it when it does not), keeps the local model sidecar in view, and
// carries its own Go-native copy of the substrate for a second opinion on
// any number. Zero external dependencies — the standard library is the whole
// supply chain.
//
//	golue run [--rust path] [--endpoint url] [--port n]   build-if-needed + serve
//	golue doctor [--endpoint url]                          lanes + endpoint health
//	golue bench [--dim n]                                  the Go-native kernel, measured
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/8b-is/alexiai/go/gaiaquant"
)

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		usage()
		os.Exit(1)
	}
	var err error
	switch args[0] {
	case "run":
		err = run(args[1:])
	case "doctor":
		err = doctor(args[1:])
	case "bench":
		err = bench(args[1:])
	case "help", "-h", "--help":
		usage()
	default:
		usage()
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "\n  %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Print(`golue — the glue between the lanes

  golue run [--rust PATH] [--endpoint URL] [--port N]   build-if-needed + serve
  golue doctor [--endpoint URL]                          lanes + endpoint health
  golue bench [--dim N]                                  the Go-native kernel, measured
`)
}

func flag(args []string, name string) (string, bool) {
	for i, a := range args {
		if a == "--"+name && i+1 < len(args) {
			return args[i+1], true
		}
	}
	return "", false
}

// rustBinary resolves the release binary, building it when missing.
func rustBinary(explicit string) (string, error) {
	if explicit != "" {
		return explicit, nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	bin := filepath.Join(cwd, "..", "rust", "target", "release", "alexiai")
	if _, err := os.Stat(bin); os.IsNotExist(err) {
		fmt.Fprintln(os.Stderr, "  (rust binary missing — cargo build --release)")
		cmd := exec.Command("cargo", "build", "--release", "-p", "alexiai")
		cmd.Dir = filepath.Join(cwd, "..", "rust")
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return "", fmt.Errorf("cargo build: %w", err)
		}
	}
	return bin, nil
}

func run(args []string) error {
	rustPath, _ := flag(args, "rust")
	endpoint, hasEndpoint := flag(args, "endpoint")
	port, hasPort := flag(args, "port")
	if !hasPort {
		port = "8787"
	}

	bin, err := rustBinary(rustPath)
	if err != nil {
		return err
	}

	cmdArgs := []string{"serve", "--port", port}
	if hasEndpoint {
		cmdArgs = append(cmdArgs, "--endpoint", endpoint)
	}
	cmd := exec.Command(bin, cmdArgs...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	fmt.Fprintf(os.Stderr, "  golue: %s serve --port %s\n", bin, port)
	return cmd.Run()
}

func doctor(args []string) error {
	endpoint, hasEndpoint := flag(args, "endpoint")
	if !hasEndpoint {
		endpoint = "http://127.0.0.1:1337"
	}
	bin, err := rustBinary("")
	if err == nil {
		fmt.Printf("  rust lane   %s\n", bin)
	} else {
		fmt.Printf("  rust lane   (not built yet)\n")
	}
	fmt.Printf("  go lane     gaiaquant %s (%.4f bits/weight)\n",
		goVersion(), gaiaquant.BitsPerWeight)
	fmt.Printf("  endpoint    %s\n", endpoint)
	fmt.Printf("  policy      inference never leaves this machine — loopback or nothing\n")
	return nil
}

func goVersion() string {
	out, err := exec.Command("go", "version").Output()
	if err != nil {
		return "unknown"
	}
	fields := strings.Fields(string(out))
	if len(fields) >= 3 {
		return fields[2]
	}
	return string(out[:len(out)-1])
}

func bench(args []string) error {
	dim := 256
	if d, ok := flag(args, "dim"); ok {
		fmt.Sscanf(d, "%d", &dim)
	}
	rows, cols := dim, dim

	// Reproducible weights: mulberry-ish, no allocations in the loop.
	w := make([]float64, rows*cols)
	s := uint32(0x5eed)
	for i := range w {
		s += 0x6d2b79f5
		t := s
		t = (t ^ (t >> 15)) * (t | 1)
		t ^= t + (t^(t>>7))*(t|61)
		w[i] = float64((t^(t>>14))>>8)/16777216*2 - 1
	}
	x := make([]float64, rows)
	for j := range x {
		x[j] = w[j]
	}

	trits, scales := gaiaquant.QuantizePerRow(w, rows, cols)
	packed := make([]uint8, (rows*cols+4)/5)
	gaiaquant.PackTrits(trits, packed)

	dense := make([]float64, cols)
	ternary := make([]float64, cols)

	start := time.Now()
	gaiaquant.DenseMatMul(w, rows, cols, x, dense)
	denseNs := time.Since(start).Nanoseconds()

	start = time.Now()
	gaiaquant.TernaryMatMul(packed, rows, cols, x, scales, 1, ternary)
	ternaryNs := time.Since(start).Nanoseconds()

	fmt.Println("  gaiaquant bench — honest numbers")
	fmt.Printf("    shape      %dx%d\n", rows, cols)
	fmt.Printf("    packing    %.4f bits/weight · %d bytes vs %d float64 (%.1fx)\n",
		gaiaquant.BitsPerWeight, len(packed), len(w)*8, float64(len(w)*8)/float64(len(packed)))
	fmt.Printf("    dense      %d ns\n", denseNs)
	fmt.Printf("    ternary    %d ns\n", ternaryNs)
	fmt.Printf("    error      %.4f relative L2 vs dense (random weights — the worst case)\n",
		gaiaquant.RelativeError(ternary, dense))
	return nil
}
