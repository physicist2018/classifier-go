package main

import (
	"classifier-go/internal/config"
	"classifier-go/pkg/convolve"
	"classifier-go/pkg/heatmapplotter"
	"classifier-go/pkg/normalboxmueller"
	"classifier-go/pkg/readmatrix"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sync"

	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

const (
	penaltyCoefficient = 1000
	convergenceTol     = 1e-5
	maxIterations      = 1000
)

func main() {
	cfg := config.Parse()
	log.Println("Starting classifier...")

	// Load input matrices
	flMatrix, err := readmatrix.ReadMatrix(filepath.Join(cfg.InputDir, "FL_cap.txt"))
	if err != nil {
		log.Fatal("Error reading Fluorescence Capacity Matrix:", err)
	}

	depMatrix, err := readmatrix.ReadMatrix(filepath.Join(cfg.InputDir, "Dep.txt"))
	if err != nil {
		log.Fatal("Error reading Depolarization Matrix:", err)
	}

	// Process matrices
	processMatrices(cfg, flMatrix, depMatrix)

	// Convert reference deltas
	deltaD, deltaU, deltaS := convertDeltas(cfg)

	// Generate parameter distributions
	gfUs, gfDs, gfSs, deltaUs, deltaDs, deltaSs := generateParameterDistributions(cfg, deltaD, deltaU, deltaS)
	saveArraysToCSV(filepath.Join(cfg.InputDir, "rand.csv"), gfUs, deltaUs, gfDs, deltaDs, gfSs, deltaSs)

	// Initialize result matrices
	resultMats := initializeResultMatrices(flMatrix)

	// Process data points and get tuned parameters
	newGf, newDelta := processDataPoints(
		cfg,
		flMatrix, depMatrix,
		gfUs, gfDs, gfSs,
		deltaUs, deltaDs, deltaSs,
		resultMats,
	)

	// Save results
	saveResults(cfg.InputDir, resultMats)

	// Generate heatmaps
	generateHeatmaps(cfg.InputDir, depMatrix, resultMats)

	// Print final results
	printFinalResults(newGf, newDelta)
}

func processMatrices(cfg *config.Config, flMatrix, depMatrix *mat.Dense) {
	rows, cols := depMatrix.Dims()
	tmpSlice := depMatrix.Slice(0, rows, 1, cols).(*mat.Dense)
	tmpSlice.Scale(0.01, tmpSlice)

	if cfg.DoSmooth {
		gs := convolve.NewGaussianKernel(float64(cfg.SigmaT), float64(cfg.SigmaH), cfg.Size)
		smoothMatrix(flMatrix, gs, rows, cols)
		smoothMatrix(depMatrix, gs, rows, cols)
	}
}

func smoothMatrix(matrix *mat.Dense, gs *convolve.ConvolveKernel, rows, cols int) {
	r := matrix.Slice(0, rows, 1, cols).(*mat.Dense)
	output := gs.Convolve(r)
	r.Copy(output)
}

func convertDeltas(cfg *config.Config) (float64, float64, float64) {
	return cfg.DeltaDust,
		cfg.DeltaUrban,
		cfg.DeltaSmoke
}

func generateParameterDistributions(cfg *config.Config, deltaD, deltaU, deltaS float64) ([]float64, []float64, []float64, []float64, []float64, []float64) {
	// Urban
	gfuMin := cfg.GfUrban * (1.0 - cfg.VariationCoefficientGf)
	gfuMax := cfg.GfUrban * (1.0 + cfg.VariationCoefficientGf)
	genGfUrban := normalboxmueller.NewNormalDistParams(cfg.GfUrban, (gfuMax-gfuMin)/4, gfuMin, gfuMax)
	gfUs := genGfUrban.RandN(cfg.NumPoints)

	// Soot
	gfsMin := cfg.GfSmoke * (1.0 - cfg.VariationCoefficientGf)
	gfsMax := cfg.GfSmoke * (1.0 + cfg.VariationCoefficientGf)
	genGfSoot := normalboxmueller.NewNormalDistParams(cfg.GfSmoke, (gfsMax-gfsMin)/4, gfsMin, gfsMax)
	gfSs := genGfSoot.RandN(cfg.NumPoints)

	// Dust
	gfdMin := cfg.GfDust * (1.0 - cfg.VariationCoefficientGf)
	gfdMax := cfg.GfDust * (1.0 + cfg.VariationCoefficientGf)
	genGfDust := normalboxmueller.NewNormalDistParams(cfg.GfDust, (gfdMax-gfdMin)/4, gfdMin, gfdMax)
	gfDs := genGfDust.RandN(cfg.NumPoints)

	// Delta distributions
	deltaSMin := deltaS * (1.0 - cfg.VariationCoefficientDelta)
	deltaSMax := deltaS * (1.0 + cfg.VariationCoefficientDelta)
	genDeltaSoot := normalboxmueller.NewNormalDistParams(deltaS, (deltaSMax-deltaSMin)/4, deltaSMin, deltaSMax)
	deltaSs := genDeltaSoot.RandN(cfg.NumPoints)

	deltaUMin := deltaU * (1.0 - cfg.VariationCoefficientDelta)
	deltaUMax := deltaU * (1.0 + cfg.VariationCoefficientDelta)
	genDeltaUrban := normalboxmueller.NewNormalDistParams(deltaU, (deltaUMax-deltaUMin)/4, deltaUMin, deltaUMax)
	deltaUs := genDeltaUrban.RandN(cfg.NumPoints)

	deltaDMin := deltaD * (1.0 - cfg.VariationCoefficientDelta)
	deltaDMax := deltaD * (1.0 + cfg.VariationCoefficientDelta)
	genDeltaDust := normalboxmueller.NewNormalDistParams(deltaD, (deltaDMax-deltaDMin)/4, deltaDMin, deltaDMax)
	deltaDs := genDeltaDust.RandN(cfg.NumPoints)

	return gfUs, gfDs, gfSs, deltaUs, deltaDs, deltaSs
}

type ResultMatrices struct {
	EtaU, EtaS, EtaD          *mat.Dense
	GfUN, GfDN, GfSN          *mat.Dense
	DeltaUN, DeltaDN, DeltaSN *mat.Dense
	ErrMat                    *mat.Dense
}

func initializeResultMatrices(flMatrix *mat.Dense) *ResultMatrices {
	r, c := flMatrix.Dims()
	mats := &ResultMatrices{
		EtaU:    mat.NewDense(r, c, nil),
		EtaS:    mat.NewDense(r, c, nil),
		EtaD:    mat.NewDense(r, c, nil),
		GfUN:    mat.NewDense(r, c, nil),
		GfDN:    mat.NewDense(r, c, nil),
		GfSN:    mat.NewDense(r, c, nil),
		DeltaUN: mat.NewDense(r, c, nil),
		DeltaDN: mat.NewDense(r, c, nil),
		DeltaSN: mat.NewDense(r, c, nil),
		ErrMat:  mat.NewDense(r, c, nil),
	}

	// Copy first column
	for i := 0; i < r; i++ {
		val := flMatrix.At(i, 0)
		mats.EtaU.Set(i, 0, val)
		mats.EtaS.Set(i, 0, val)
		mats.EtaD.Set(i, 0, val)
		mats.GfUN.Set(i, 0, val)
		mats.GfDN.Set(i, 0, val)
		mats.GfSN.Set(i, 0, val)
		mats.DeltaUN.Set(i, 0, val)
		mats.DeltaDN.Set(i, 0, val)
		mats.DeltaSN.Set(i, 0, val)
		mats.ErrMat.Set(i, 0, val)
	}

	return mats
}

func processDataPoints(
	cfg *config.Config,
	flMatrix, depMatrix *mat.Dense,
	gfUs, gfDs, gfSs, deltaUs, deltaDs, deltaSs []float64,
	resultMats *ResultMatrices,
) ([3]float64, [3]float64) {

	numWorkers := runtime.NumCPU()
	taskQueue := make(chan task, cfg.NumPoints)
	resultQueue := make(chan result, cfg.NumPoints)

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go worker(taskQueue, resultQueue, &wg)
	}

	r, c := flMatrix.Dims()
	var (
		newGf      [3]float64
		newDelta   [3]float64
		pointCount int
	)

	for i := 0; i < r; i++ {
		fmt.Print("\033[1A")
		fmt.Print("\033[K")
		fmt.Printf("Processing row %d/%d\n", i+1, r)
		for j := 1; j < c; j++ {
			gfMeas := flMatrix.At(i, j)
			deltaMeas := depMatrix.At(i, j)

			// Skip invalid points
			if gfMeas <= 0 || deltaMeas <= 0 || math.IsNaN(gfMeas) || math.IsNaN(deltaMeas) {
				continue
			}

			// Process point
			etasMean := processPoint(
				cfg, gfMeas, deltaMeas,
				gfUs, gfDs, gfSs,
				deltaUs, deltaDs, deltaSs,
				taskQueue, resultQueue,
			)

			if etasMean == nil {
				continue
			}

			// Update result matrices
			resultMats.EtaU.Set(i, j, etasMean[0].X)
			resultMats.EtaD.Set(i, j, etasMean[1].X)
			resultMats.EtaS.Set(i, j, etasMean[2].X)

			resultMats.GfUN.Set(i, j, etasMean[0].Gf)
			resultMats.GfDN.Set(i, j, etasMean[1].Gf)
			resultMats.GfSN.Set(i, j, etasMean[2].Gf)

			resultMats.DeltaUN.Set(i, j, etasMean[0].Delta)
			resultMats.DeltaDN.Set(i, j, etasMean[1].Delta)
			resultMats.DeltaSN.Set(i, j, etasMean[2].Delta)
			resultMats.ErrMat.Set(i, j, etasMean[0].F)

			// Accumulate for averages
			newGf[0] += etasMean[0].Gf
			newGf[1] += etasMean[1].Gf
			newGf[2] += etasMean[2].Gf

			newDelta[0] += etasMean[0].Delta
			newDelta[1] += etasMean[1].Delta
			newDelta[2] += etasMean[2].Delta

			pointCount++
		}
	}

	close(taskQueue)
	wg.Wait()

	// Calculate final averages
	if pointCount > 0 {
		for i := 0; i < 3; i++ {
			newGf[i] /= float64(pointCount)
			newDelta[i] /= float64(pointCount)
		}
	}

	return newGf, newDelta
}

func processPoint(
	cfg *config.Config,
	gfMeas, deltaMeas float64,
	gfUs, gfDs, gfSs, deltaUs, deltaDs, deltaSs []float64,
	taskQueue chan task, resultQueue chan result,
) []avgresult {

	// Send tasks to workers
	for k := 0; k < cfg.NumPoints; k++ {
		taskQueue <- task{
			GF_meas:    gfMeas,
			delta_meas: deltaMeas,
			GF_u_k:     gfUs[k],
			GF_d_k:     gfDs[k],
			GF_s_k:     gfSs[k],
			delta_u_k:  deltaUs[k],
			delta_d_k:  deltaDs[k],
			delta_s_k:  deltaSs[k],
		}
	}

	// Collect results
	var tmpEta []result
	for k := 0; k < cfg.NumPoints; k++ {
		res := <-resultQueue
		if res.Valid {
			tmpEta = append(tmpEta, res)
		}
	}

	if len(tmpEta) == 0 {
		return nil
	}

	return averageVectors(tmpEta, 0.1)
}

type task struct {
	GF_meas    float64
	delta_meas float64
	GF_u_k     float64
	GF_d_k     float64
	GF_s_k     float64
	delta_u_k  float64
	delta_d_k  float64
	delta_s_k  float64
}

type result struct {
	X     []float64
	F     float64
	Gf    [3]float64
	Delta [3]float64
	Valid bool
}

type avgresult struct {
	X     float64
	F     float64
	Delta float64
	Gf    float64
}

func worker(tasks <-chan task, results chan<- result, wg *sync.WaitGroup) {
	defer wg.Done()
	for t := range tasks {
		ntas, F, err := classifySinglePoint(
			t.GF_meas,
			t.delta_meas,
			t.GF_u_k,
			t.GF_d_k,
			t.GF_s_k,
			t.delta_u_k,
			t.delta_d_k,
			t.delta_s_k,
		)

		if err == nil && isValidSolution(ntas) {
			results <- result{
				X:     ntas,
				F:     F,
				Gf:    [3]float64{t.GF_u_k, t.GF_d_k, t.GF_s_k},
				Delta: [3]float64{t.delta_u_k, t.delta_d_k, t.delta_s_k},
				Valid: true,
			}
		} else {
			results <- result{Valid: false}
		}
	}
}

func classifySinglePoint(GFMeas, deltaMeas, GFU, GFD, GFS, deltaU, deltaD, deltaS float64) ([]float64, float64, error) {
	objectiveFunction := func(x []float64) float64 {
		nu, nd, ns := x[0], x[1], x[2]

		residuals := []float64{
			1 - nu - ns - nd,
			(GFMeas - (ns*GFS + nu*GFU + nd*GFD)),
			(deltaMeas - (ns*deltaS + nu*deltaU + nd*deltaD)),
		}

		penalty := calculatePenalty(nu, nd, ns)
		for i := range residuals {
			residuals[i] += penalty
		}

		return math.Pow(floats.Norm(residuals, 2), 2)
	}

	objectiveGrad := func(grad, x []float64) {
		fd.Gradient(grad, objectiveFunction, x, nil)
	}

	problem := optimize.Problem{
		Grad: objectiveGrad,
		Func: objectiveFunction,
	}

	method := &optimize.BFGS{}
	settings := &optimize.Settings{
		MajorIterations: maxIterations,
		FuncEvaluations: 1000,
		Converger: &optimize.FunctionConverge{
			Relative:   convergenceTol,
			Absolute:   convergenceTol,
			Iterations: 1000,
		},
	}

	initialGuess := []float64{0.1, 0.5, 0.4} // Более сбалансированное начальное приближение
	result, err := optimize.Minimize(problem, initialGuess, settings, method)
	if err != nil {
		return nil, 0, fmt.Errorf("optimization failed: %v", err)
	}

	// Apply bounds
	for i := range result.X {
		result.X[i] = math.Max(0, math.Min(1, result.X[i]))
	}

	// Normalize to sum to 1
	sum := floats.Sum(result.X)
	if sum > 0 {
		floats.Scale(1/sum, result.X)
	}

	if math.Abs(sum-1) > 0.05 { // Более строгая проверка
		return nil, 0, fmt.Errorf("invalid solution: sum of parameters = %f", sum)
	}

	return result.X, result.F, nil
}

func calculatePenalty(nu, nd, ns float64) float64 {
	penalty := 0.0
	addPenalty := func(val float64) {
		if val < 0 {
			penalty += penaltyCoefficient * val * val
		} else if val > 1 {
			penalty += penaltyCoefficient * (val - 1) * (val - 1)
		}
	}

	addPenalty(nu)
	addPenalty(nd)
	addPenalty(ns)

	return penalty
}

func isValidSolution(x []float64) bool {
	for _, v := range x {
		if v < 0 || v > 1 || math.IsNaN(v) {
			return false
		}
	}

	return floats.Sum(x) > 0.95 && floats.Sum(x) < 1.05 // Допуск 5%
}

func averageVectors(vectors []result, avgFrac float64) []avgresult {
	if len(vectors) == 0 {
		return nil
	}

	// Сортируем по возрастанию значения функции (лучшие решения сначала)
	slices.SortFunc(vectors, func(a, b result) int {
		if a.F < b.F {
			return -1
		} else if a.F > b.F {
			return 1
		}
		return 0
	})

	N := int(float64(len(vectors)) * avgFrac)
	if N < 1 {
		N = 1
	}

	sum := make([]avgresult, 3)
	for i := 0; i < N; i++ {
		for j := 0; j < 3; j++ {
			sum[j].F += vectors[i].F
			sum[j].X += vectors[i].X[j]
			sum[j].Gf += vectors[i].Gf[j]
			sum[j].Delta += vectors[i].Delta[j]
		}
	}

	// Усредняем
	for j := 0; j < 3; j++ {
		sum[j].F /= float64(N)
		sum[j].X /= float64(N)
		sum[j].Gf /= float64(N)
		sum[j].Delta /= float64(N)
	}

	return sum
}

func saveResults(outputDir string, m *ResultMatrices) {
	save := func(name string, matrix *mat.Dense) {
		if err := saveMatrix(filepath.Join(outputDir, name), matrix); err != nil {
			log.Printf("Error saving %s: %v", name, err)
		}
	}

	save("Eta_u.csv", m.EtaU)
	save("Eta_d.csv", m.EtaD)
	save("Eta_s.csv", m.EtaS)
	save("Gf_u.csv", m.GfUN)
	save("Gf_d.csv", m.GfDN)
	save("Gf_s.csv", m.GfSN)
	save("Delta_u.csv", m.DeltaUN)
	save("Delta_d.csv", m.DeltaDN)
	save("Delta_s.csv", m.DeltaSN)
	save("Err.csv", m.ErrMat)
}

func saveMatrix(filename string, m mat.Matrix) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if j > 0 {
				f.WriteString("\t")
			}
			fmt.Fprintf(f, "%.6e", m.At(i, j))
		}
		f.WriteString("\n")
	}
	return nil
}

func generateHeatmaps(outputDir string, depMatrix *mat.Dense, m *ResultMatrices) {
	plot := func(name, path string, matrix *mat.Dense) {
		heatmapplotter.MakeHeatmapPlot(matrix, name, path)
	}

	plot("Depolarization", filepath.Join(outputDir, "Dep.pdf"), depMatrix)
	plot("Eta Urban", filepath.Join(outputDir, "Eta_u.pdf"), m.EtaU)
	plot("Eta Dust", filepath.Join(outputDir, "Eta_d.pdf"), m.EtaD)
	plot("Eta Soot", filepath.Join(outputDir, "Eta_s.pdf"), m.EtaS)
}

func printFinalResults(newGf, newDelta [3]float64) {
	fmt.Println("\n=== Final Tuned Parameters ===")
	fmt.Printf("Urban aerosol: Gf = %.3e, δ = %.3f\n", newGf[0], newDelta[0])
	fmt.Printf("Dust  aerosol: Gf = %.3e, δ = %.3f\n", newGf[1], newDelta[1])
	fmt.Printf("Soot  aerosol: Gf = %.3e, δ = %.3f\n", newGf[2], newDelta[2])
}

func saveArraysToCSV(filename string, arrs ...[]float64) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write headers
	headers := make([]string, len(arrs))
	for i := range headers {
		headers[i] = fmt.Sprintf("Array%d", i+1)
	}
	if err := writer.Write(headers); err != nil {
		return err
	}

	// Write data
	maxLength := maxSliceLength(arrs...)
	for i := 0; i < maxLength; i++ {
		row := make([]string, len(arrs))
		for j, arr := range arrs {
			if i < len(arr) {
				row[j] = fmt.Sprintf("%.6e", arr[i])
			}
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return nil
}

func maxSliceLength(slices ...[]float64) int {
	max := 0
	for _, s := range slices {
		if len(s) > max {
			max = len(s)
		}
	}
	return max
}
