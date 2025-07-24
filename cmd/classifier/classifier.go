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
	"runtime"
	"slices"
	"sync"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

// Constants
const (
	penaltyCoefficient = 1000
	convergenceTol     = 1e-4
	maxIterations      = 100
)

func main() {
	// Load configuration
	cfg := config.Parse()
	log.Println("Starting classifier...")

	// Load and process input matrices
	fluorescenceCapacityMatrix, err := readmatrix.ReadMatrix(cfg.InputDir + "FL_cap.txt")
	if err != nil {
		log.Fatal("Error reading Fluorescence Capacity Matrix:", err)
	}

	depolarizationMatrix, err := readmatrix.ReadMatrix(cfg.InputDir + "Dep.txt")
	if err != nil {
		log.Fatal("Error reading Depolarization Matrix:", err)
	}

	// Scale and smooth matrices if needed
	processMatrices(cfg, fluorescenceCapacityMatrix, depolarizationMatrix)

	// Convert reference depolarization values
	deltaD, deltaU, deltaS := convertDeltas(cfg)

	// Generate random parameter distributions
	gfUs, gfDs, gfSs, deltaUs, deltaDs, deltaSs := generateParameterDistributions(cfg, deltaD, deltaU, deltaS)
	saveArraysToCSV("rand.csv", gfUs, deltaUs, gfDs, deltaDs, gfSs, deltaSs)

	// Initialize result matrices
	resultMatrices := initializeResultMatrices(fluorescenceCapacityMatrix)

	// Process data points
	newGf, newDelta := processDataPoints(
		cfg,
		fluorescenceCapacityMatrix,
		depolarizationMatrix,
		gfUs, gfDs, gfSs,
		deltaUs, deltaDs, deltaSs,
		resultMatrices,
	)

	// Calculate averages
	calculateAverages(newGf, newDelta, fluorescenceCapacityMatrix)

	// Save results
	saveResults(cfg.InputDir, resultMatrices)

	// Generate heatmaps
	generateHeatmaps(cfg.InputDir, depolarizationMatrix, resultMatrices)

	// Print final results
	printFinalResults(newGf, newDelta)
}

// Helper functions

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

func convertDeltas(cfg *config.Config) (deltaD, deltaU, deltaS float64) {
	deltaD = cfg.DeltaDust / (1.0 + cfg.DeltaDust)
	deltaU = cfg.DeltaUrban / (1.0 + cfg.DeltaUrban)
	deltaS = cfg.DeltaSoot / (1.0 + cfg.DeltaSoot)
	return
}

func generateParameterDistributions(cfg *config.Config, deltaD, deltaU, deltaS float64) (
	gfUs, gfDs, gfSs, deltaUs, deltaDs, deltaSs []float64,
) {
	// Urban aerosol
	gfuMin := cfg.GfUrban * (1.0 - cfg.VariationCoefficient)
	gfuMax := cfg.GfUrban * (1.0 + cfg.VariationCoefficient)
	genGfUrban := normalboxmueller.NewNormalDistParams(
		cfg.GfUrban, (gfuMax-gfuMin)/4, gfuMin, gfuMax,
	)
	gfUs = genGfUrban.RandN(cfg.NumPoints)

	// Soot
	gfsMin := cfg.GfSoot * (1.0 - cfg.VariationCoefficient)
	gfsMax := cfg.GfSoot * (1.0 + cfg.VariationCoefficient)
	genGfSoot := normalboxmueller.NewNormalDistParams(
		cfg.GfSoot, (gfsMax-gfsMin)/4, gfsMin, gfsMax,
	)
	gfSs = genGfSoot.RandN(cfg.NumPoints)

	// Dust
	gfdMin := cfg.GfDust * (1.0 - cfg.VariationCoefficient)
	gfdMax := cfg.GfDust * (1.0 + cfg.VariationCoefficient)
	genGfDust := normalboxmueller.NewNormalDistParams(
		cfg.GfDust, (gfdMax-gfdMin)/4, gfdMin, gfdMax,
	)
	gfDs = genGfDust.RandN(cfg.NumPoints)

	// Delta distributions
	deltaSMin := deltaS * (1.0 - cfg.VariationCoefficient)
	deltaSMax := deltaS * (1.0 + cfg.VariationCoefficient)
	genDeltaSoot := normalboxmueller.NewNormalDistParams(
		deltaS, (deltaSMax-deltaSMin)/4, deltaSMin, deltaSMax,
	)
	deltaSs = genDeltaSoot.RandN(cfg.NumPoints)

	deltaUMin := deltaU * (1.0 - cfg.VariationCoefficient)
	deltaUMax := deltaU * (1.0 + cfg.VariationCoefficient)
	genDeltaUrban := normalboxmueller.NewNormalDistParams(
		deltaU, (deltaUMax-deltaUMin)/4, deltaUMin, deltaUMax,
	)
	deltaUs = genDeltaUrban.RandN(cfg.NumPoints)

	deltaDMin := deltaD * (1.0 - cfg.VariationCoefficient)
	deltaDMax := deltaD * (1.0 + cfg.VariationCoefficient)
	genDeltaDust := normalboxmueller.NewNormalDistParams(
		deltaD, (deltaDMax-deltaDMin)/4, deltaDMin, deltaDMax,
	)
	deltaDs = genDeltaDust.RandN(cfg.NumPoints)

	return gfUs, gfDs, gfSs, deltaUs, deltaDs, deltaSs
}

type ResultMatrices struct {
	EtaU, EtaS, EtaD          *mat.Dense
	GfUN, GfDN, GfSN          *mat.Dense
	DeltaUN, DeltaDN, DeltaSN *mat.Dense
}

func initializeResultMatrices(flMatrix *mat.Dense) *ResultMatrices {
	r, c := flMatrix.Dims()

	matrices := &ResultMatrices{
		EtaU:    mat.NewDense(r, c, nil),
		EtaS:    mat.NewDense(r, c, nil),
		EtaD:    mat.NewDense(r, c, nil),
		GfUN:    mat.NewDense(r, c, nil),
		GfDN:    mat.NewDense(r, c, nil),
		GfSN:    mat.NewDense(r, c, nil),
		DeltaUN: mat.NewDense(r, c, nil),
		DeltaDN: mat.NewDense(r, c, nil),
		DeltaSN: mat.NewDense(r, c, nil),
	}

	// Copy first column (time/altitude)
	for i := 0; i < r; i++ {
		val := flMatrix.At(i, 0)
		matrices.EtaU.Set(i, 0, val)
		matrices.EtaS.Set(i, 0, val)
		matrices.EtaD.Set(i, 0, val)
		matrices.GfUN.Set(i, 0, val)
		matrices.GfDN.Set(i, 0, val)
		matrices.GfSN.Set(i, 0, val)
		matrices.DeltaUN.Set(i, 0, val)
		matrices.DeltaDN.Set(i, 0, val)
		matrices.DeltaSN.Set(i, 0, val)
	}

	return matrices
}

func processDataPoints(
	cfg *config.Config,
	flMatrix, depMatrix *mat.Dense,
	gfUs, gfDs, gfSs, deltaUs, deltaDs, deltaSs []float64,
	resultMatrices *ResultMatrices,
) (newGf, newDelta [3]float64) {

	numWorkers := runtime.NumCPU()
	taskQueue := make(chan task, cfg.NumPoints)
	resultQueue := make(chan result, cfg.NumPoints)

	var wg sync.WaitGroup
	for range numWorkers {
		wg.Add(1)
		go worker(taskQueue, resultQueue, &wg)
	}

	r, c := flMatrix.Dims()
	for i := 0; i < r; i++ {
		fmt.Printf("Processing row %d/%d\n", i+1, r)
		for j := 1; j < c; j++ {
			deltaMeas := depMatrix.At(i, j)
			gfMeas := flMatrix.At(i, j)

			// Process point with workers
			etasMean := processPoint(
				cfg, gfMeas, deltaMeas,
				gfUs, gfDs, gfSs,
				deltaUs, deltaDs, deltaSs,
				taskQueue, resultQueue,
			)

			if etasMean == nil {
				fmt.Printf("Warning: no valid solutions for point (%d,%d)\n", i, j)
				continue
			}

			// Store results
			updateResultMatrices(resultMatrices, i, j, etasMean)
			updateAverages(&newGf, &newDelta, etasMean)
		}
	}

	close(taskQueue)
	wg.Wait()
	return
}

func processPoint(
	cfg *config.Config,
	gfMeas, deltaMeas float64,
	gfUs, gfDs, gfSs, deltaUs, deltaDs, deltaSs []float64,
	taskQueue chan task, resultQueue chan result,
) []avgresult {
	// Send tasks to workers
	for k := range cfg.NumPoints {
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
	tmpEta := make([]result, 0, cfg.NumPoints)
	for k := 0; k < cfg.NumPoints; k++ {
		res := <-resultQueue
		if res.Valid {
			tmpEta = append(tmpEta, res)
		}
	}

	if len(tmpEta) == 0 {
		return nil
	}

	return averageVectors(tmpEta, cfg.AvgPercent)
}

func updateResultMatrices(m *ResultMatrices, i, j int, etasMean []avgresult) {
	m.EtaU.Set(i, j, etasMean[0].X)
	m.EtaD.Set(i, j, etasMean[1].X)
	m.EtaS.Set(i, j, etasMean[2].X)

	m.DeltaUN.Set(i, j, etasMean[0].Delta)
	m.DeltaDN.Set(i, j, etasMean[1].Delta)
	m.DeltaSN.Set(i, j, etasMean[2].Delta)
	m.GfUN.Set(i, j, etasMean[0].Gf)
	m.GfDN.Set(i, j, etasMean[1].Gf)
	m.GfSN.Set(i, j, etasMean[2].Gf)
}

func updateAverages(newGf, newDelta *[3]float64, etasMean []avgresult) {
	newGf[0] += etasMean[0].Gf
	newGf[1] += etasMean[1].Gf
	newGf[2] += etasMean[2].Gf

	newDelta[0] += (etasMean[0].Delta / (1 - etasMean[0].Delta))
	newDelta[1] += (etasMean[1].Delta / (1 - etasMean[1].Delta))
	newDelta[2] += (etasMean[2].Delta / (1 - etasMean[2].Delta))
}

func calculateAverages(newGf, newDelta [3]float64, flMatrix *mat.Dense) {
	r, c := flMatrix.Dims()
	nAvg := r * (c - 1)
	for i := range newGf {
		newGf[i] /= float64(nAvg)
		newDelta[i] /= float64(nAvg)
	}
}

func saveResults(outputDir string, m *ResultMatrices) {
	saveMatrix := func(filename string, matrix *mat.Dense) {
		if err := saveMatrix(outputDir+filename, matrix); err != nil {
			log.Printf("Error saving %s: %v", filename, err)
		}
	}

	saveMatrix("Eta_s.csv", m.EtaS)
	saveMatrix("Eta_u.csv", m.EtaU)
	saveMatrix("Eta_d.csv", m.EtaD)
	saveMatrix("Gf_u.csv", m.GfUN)
	saveMatrix("Gf_d.csv", m.GfDN)
	saveMatrix("Gf_s.csv", m.GfSN)
	saveMatrix("Delta_u.csv", m.DeltaUN)
	saveMatrix("Delta_d.csv", m.DeltaDN)
	saveMatrix("Delta_s.csv", m.DeltaSN)
}

func generateHeatmaps(outputDir string, depMatrix *mat.Dense, m *ResultMatrices) {
	heatmapplotter.MakeHeatmapPlot(depMatrix, "Dep", outputDir+"Dep.pdf")
	heatmapplotter.MakeHeatmapPlot(m.EtaD, "Eta_d", outputDir+"Eta_d.pdf")
	heatmapplotter.MakeHeatmapPlot(m.EtaU, "Eta_u", outputDir+"Eta_u.pdf")
	heatmapplotter.MakeHeatmapPlot(m.EtaS, "Eta_s", outputDir+"Eta_s.pdf")
}

func printFinalResults(newGf, newDelta [3]float64) {
	fmt.Println("Tuned parameters:")
	fmt.Printf("Gf_u: %.3e, delta_u: %.3f\n", newGf[0], newDelta[0])
	fmt.Printf("Gf_d: %.3e, delta_d: %.3f\n", newGf[1], newDelta[1])
	fmt.Printf("Gf_s: %.3e, delta_s: %.3f\n", newGf[2], newDelta[2])
}

// Optimization and worker-related functions

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
	Delta float64
	Gf    float64
}

func worker(tasks <-chan task, results chan<- result, wg *sync.WaitGroup) {
	defer wg.Done()
	for t := range tasks {
		ntas, F, err := classifySinglePoint(
			t.GF_meas,
			t.delta_meas/(1+t.delta_meas),
			t.GF_u_k, t.GF_d_k, t.GF_s_k,
			t.delta_u_k, t.delta_d_k, t.delta_s_k,
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
	residual := func(x []float64) []float64 {
		nu, nd, ns := x[0], x[1], x[2]

		residuals := []float64{
			1 - nu - ns - nd,
			GFMeas - (ns*GFS + nu*GFU + nd*GFD),
			deltaMeas - (ns*deltaS + nu*deltaU + nd*deltaD),
		}

		penalty := calculatePenalty(nu, nd, ns)
		for i := range residuals {
			residuals[i] += penalty
		}

		return residuals
	}

	problem := optimize.Problem{
		Func: func(x []float64) float64 {
			res := residual(x)
			return floats.Norm(res, 2)
		},
	}

	method := &optimize.NelderMead{}
	settings := &optimize.Settings{
		MajorIterations: maxIterations,
		FuncEvaluations: 1000,
		Converger: &optimize.FunctionConverge{
			Relative:   convergenceTol,
			Absolute:   convergenceTol,
			Iterations: 1000,
		},
	}

	initialGuess := []float64{0.5, 0.5, 0.5}
	result, err := optimize.Minimize(problem, initialGuess, settings, method)
	if err != nil {
		return nil, 0, fmt.Errorf("optimization failed: %v", err)
	}

	// Apply bounds
	for i := range result.X {
		result.X[i] = math.Max(0, math.Min(1, result.X[i]))
	}

	// Check sum constraint
	if math.Abs(floats.Sum(result.X)-1) > 0.01 {
		return nil, 0, fmt.Errorf("invalid solution: sum of parameters = %f", floats.Sum(result.X))
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
		if v < 0 || v > 1 {
			return false
		}
	}
	return true
}

func averageVectors(vectors []result, avgFrac float64) []avgresult {
	if len(vectors) == 0 {
		return nil
	}

	slices.SortFunc(vectors, func(a, b result) int {
		if a.F < b.F {
			return -1
		} else if a.F > b.F {
			return 1
		}
		return 0
	})

	sum := make([]avgresult, 3)
	Ntot := int(float64(len(vectors)) * avgFrac)

	for i := 0; i < Ntot; i++ {
		for j := range vectors[i].X {
			sum[j].X += vectors[i].X[j]
			sum[j].Gf += vectors[i].Gf[j]
			sum[j].Delta += vectors[i].Delta[j]
		}
	}

	for i := range sum {
		sum[i].Delta /= float64(Ntot)
		sum[i].Gf /= float64(Ntot)
		sum[i].X /= float64(Ntot)
	}

	return sum
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
			fmt.Fprintf(f, "%.4e", m.At(i, j))
		}
		f.WriteString("\n")
	}

	return nil
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
				row[j] = fmt.Sprintf("%v", arr[i])
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
