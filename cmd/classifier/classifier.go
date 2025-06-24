package main

import (
	"classifier-go/internal/config"
	"classifier-go/pkg/convolve"
	"classifier-go/pkg/heatmapplotter"
	"classifier-go/pkg/randomnormal"
	"classifier-go/pkg/readmatrix"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"sync"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

func main() {
	cfg := config.Parse()

	log.Println("Starting...")

	fluorescenceCapacityMatrix, err := readmatrix.ReadMatrix(cfg.InputDir + "FL_cap.txt")
	if err != nil {
		log.Fatal("Error reading Fluorescence Capacity Matrix", err)
	}
	depolarizationMatrix, err := readmatrix.ReadMatrix(cfg.InputDir + "Dep.txt")
	if err != nil {
		log.Fatal("Error reading depolarization Matrix", err)
	}
	rows, cols := depolarizationMatrix.Dims()
	tmpSlice := depolarizationMatrix.Slice(0, rows, 1, cols).(*mat.Dense)
	tmpSlice.Scale(0.01, tmpSlice)

	if cfg.DoSmooth {
		gs := convolve.NewGaussianKernel(float64(cfg.SigmaT), float64(cfg.SigmaH), cfg.Size)
		r := depolarizationMatrix.Slice(0, rows, 1, cols).(*mat.Dense)
		output := gs.Convolve(r)
		r.Copy(output)

		r = fluorescenceCapacityMatrix.Slice(0, rows, 1, cols).(*mat.Dense)
		output = gs.Convolve(r)
		r.Copy(output)
	}

	delta_d := cfg.DeltaDust / (1.0 + cfg.DeltaDust)
	delta_u := cfg.DeltaUrban / (1.0 + cfg.DeltaUrban)
	delta_s := cfg.DeltaSoot / (1.0 + cfg.DeltaSoot)

	// Generate random values
	gfuMin := cfg.GfUrban * (1.0 - cfg.VariationCoefficient)
	gfuMax := cfg.GfUrban * (1.0 + cfg.VariationCoefficient)
	genGfUrban := randomnormal.NewNormalRandGenerator(cfg.GfUrban, (gfuMax-gfuMin)/6, gfuMin, gfuMax)
	gfUs := genGfUrban.RandN(cfg.NumPoints)

	gfsMin := cfg.GfSoot * (1.0 - cfg.VariationCoefficient)
	gfsMax := cfg.GfSoot * (1.0 + cfg.VariationCoefficient)
	genGfSoot := randomnormal.NewNormalRandGenerator(cfg.GfSoot, (gfsMax-gfsMin)/6, gfsMin, gfsMax)
	gfSs := genGfSoot.RandN(cfg.NumPoints)

	gfdMin := cfg.GfDust * (1.0 - cfg.VariationCoefficient)
	gfdMax := cfg.GfDust * (1.0 + cfg.VariationCoefficient)
	genGfDust := randomnormal.NewNormalRandGenerator(cfg.GfDust, (gfdMax-gfdMin)/6, gfdMin, gfdMax)
	gfDs := genGfDust.RandN(cfg.NumPoints)

	delta_s_min := delta_s * (1.0 - cfg.VariationCoefficient)
	delta_s_max := delta_s * (1.0 + cfg.VariationCoefficient)
	genDeltaSoot := randomnormal.NewNormalRandGenerator(delta_s, (delta_s_max-delta_s_min)/6, delta_s_min, delta_s_max)
	delta_ss := genDeltaSoot.RandN(cfg.NumPoints)

	delta_u_min := delta_u * (1.0 - cfg.VariationCoefficient)
	delta_u_max := delta_u * (1.0 + cfg.VariationCoefficient)
	genDeltaUrban := randomnormal.NewNormalRandGenerator(delta_u, (delta_u_max-delta_u_min)/6, delta_u_min, delta_u_max)
	delta_us := genDeltaUrban.RandN(cfg.NumPoints)

	delta_d_min := delta_d * (1.0 - cfg.VariationCoefficient)
	delta_d_max := delta_d * (1.0 + cfg.VariationCoefficient)
	genDeltaDust := randomnormal.NewNormalRandGenerator(delta_d, (delta_d_max-delta_d_min)/6, delta_d_min, delta_d_max)
	delta_ds := genDeltaDust.RandN(cfg.NumPoints)

	r, c := fluorescenceCapacityMatrix.Dims()
	if r == 0 || c == 0 {
		fmt.Println("No data in FL matrix")
		return
	}

	// Initialize result matrices
	Eta_u := mat.NewDense(r, c, nil)
	Eta_s := mat.NewDense(r, c, nil)
	Eta_d := mat.NewDense(r, c, nil)

	// Copy first column (time/altitude)
	for i := 0; i < r; i++ {
		Eta_u.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Eta_s.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Eta_d.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
	}

	// Создаем пул воркеров
	numWorkers := runtime.NumCPU() // Используем все доступные ядра
	taskQueue := make(chan task, cfg.NumPoints)
	resultQueue := make(chan []float64, cfg.NumPoints)

	// Запускаем воркеры
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go worker(taskQueue, resultQueue, &wg)
	}

	// Main processing loop
	for i := 0; i < r; i++ {
		fmt.Printf("Row = %d\n", i)
		for j := 1; j < c; j++ {
			delta_meas := depolarizationMatrix.At(i, j) // 100.0
			GF_meas := fluorescenceCapacityMatrix.At(i, j)

			// Отправляем задачи воркерам
			for k := 0; k < cfg.NumPoints; k++ {
				taskQueue <- task{
					GF_meas:    GF_meas,
					delta_meas: delta_meas,
					GF_u_k:     gfUs[k],
					GF_d_k:     gfDs[k],
					GF_s_k:     gfSs[k],
					delta_u_k:  delta_us[k],
					delta_d_k:  delta_ds[k],
					delta_s_k:  delta_ss[k],
				}
			}

			// Собираем результаты
			tmp_eta := make([][]float64, 0, cfg.NumPoints)
			for k := 0; k < cfg.NumPoints; k++ {
				res := <-resultQueue
				if res != nil { // nil означает недопустимое решение
					tmp_eta = append(tmp_eta, res)
				}
			}

			if len(tmp_eta) == 0 {
				fmt.Printf("Warning: no valid solutions for point (%d,%d)\n", i, j)
				continue
			}

			// Average the valid estimates
			etas_mean := averageVectors(tmp_eta)
			Eta_u.Set(i, j, etas_mean[0])
			Eta_d.Set(i, j, etas_mean[1])
			Eta_s.Set(i, j, etas_mean[2])
		}
	}

	// Завершаем работу воркеров
	close(taskQueue)
	wg.Wait()

	err = saveMatrix(cfg.InputDir+"Eta_s.csv", Eta_s)
	if err != nil {
		log.Fatal("Error saving Eta_s matrix", err)
	}

	err = saveMatrix(cfg.InputDir+"Eta_u.csv", Eta_u)
	if err != nil {
		log.Fatal("Error saving Eta_u matrix", err)
	}

	err = saveMatrix(cfg.InputDir+"Eta_d.csv", Eta_d)
	if err != nil {
		log.Fatal("Error saving Eta_d matrix", err)
	}
	heatmapplotter.MakeHeatmapPlot(depolarizationMatrix, "Dep", cfg.InputDir+"Dep.pdf")
	heatmapplotter.MakeHeatmapPlot(Eta_d, "Eta_d", cfg.InputDir+"Eta_d.pdf")
	heatmapplotter.MakeHeatmapPlot(Eta_u, "Eta_u", cfg.InputDir+"Eta_u.pdf")
	heatmapplotter.MakeHeatmapPlot(Eta_s, "Eta_s", cfg.InputDir+"Eta_s.pdf")
	//_ = fluorescenceCapacityMatrix
	//_ = depolarizationMatrix
}

// Структура задачи для воркера
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

// Функция воркера
func worker(tasks <-chan task, results chan<- []float64, wg *sync.WaitGroup) {
	defer wg.Done()
	for t := range tasks {
		ntas_i, err := classifySinglePoint(t.GF_meas, t.delta_meas/(1+t.delta_meas),
			t.GF_u_k, t.GF_d_k, t.GF_s_k, t.delta_u_k, t.delta_d_k, t.delta_s_k)

		if err == nil && isValidSolution(ntas_i) {
			results <- ntas_i
		} else {
			results <- nil
		}
	}
}

// Проверка, что решение находится в допустимых пределах [0;1]
func isValidSolution(x []float64) bool {
	for _, v := range x {
		if v < 0 || v > 1 {
			return false
		}
	}
	return true
}

func classifySinglePoint(GF_meas, delta_meas, GF_u, GF_d, GF_s, delta_u, delta_d, delta_s float64) ([]float64, error) {
	residual := func(x []float64) []float64 {
		nu, nd, ns := x[0], x[1], x[2]

		// Базовые остатки
		residuals := []float64{
			1 - nu - ns - nd,
			GF_meas - (ns*GF_s + nu*GF_u + nd*GF_d),
			delta_meas - (ns*delta_s + nu*delta_u + nd*delta_d),
		}

		// Квадратичные штрафы за выход за границы [0, 1]
		penalty := 0.0

		// Штраф для nu ∉ [0, 1]
		if nu < 0 {
			penalty += 1000 * nu * nu
		} else if nu > 1 {
			penalty += 1000 * (nu - 1) * (nu - 1)
		}

		// Штраф для nd ∉ [0, 1]
		if nd < 0 {
			penalty += 1000 * nd * nd
		} else if nd > 1 {
			penalty += 1000 * (nd - 1) * (nd - 1)
		}

		// Штраф для ns ∉ [0, 1]
		if ns < 0 {
			penalty += 1000 * ns * ns
		} else if ns > 1 {
			penalty += 1000 * (ns - 1) * (ns - 1)
		}

		// Добавляем штраф ко всем компонентам остатка
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

	// Use NelderMead optimization method
	method := &optimize.NelderMead{}

	// Add explicit bounds [0, 1] for all parameters
	settings := &optimize.Settings{
		MajorIterations: 100,
		FuncEvaluations: 1000,
		Converger: &optimize.FunctionConverge{
			Relative:   1e-4,
			Absolute:   1e-4,
			Iterations: 1000,
		},
	}

	initialGuess := []float64{0.5, 0.5, 0.5}
	result, err := optimize.Minimize(problem, initialGuess, settings, method)
	//fmt.Println(result.Stats.FuncEvaluations)
	if err != nil {
		return nil, fmt.Errorf("optimization failed: %v", err)
	}

	// Дополнительная проверка границ
	for i := range result.X {
		result.X[i] = math.Max(0, math.Min(1, result.X[i]))
	}

	// Проверка суммы параметров (должна быть близка к 1)
	sum := floats.Sum(result.X)
	if math.Abs(sum-1) > 0.1 { // Допустимое отклонение 10%
		return nil, fmt.Errorf("invalid solution: sum of parameters = %f", sum)
	}

	return result.X, nil
}

func averageVectors(vectors [][]float64) []float64 {
	if len(vectors) == 0 {
		return []float64{0, 0, 0}
	}

	sum := make([]float64, 3)
	for _, v := range vectors {
		for i := range v {
			sum[i] += v[i]
		}
	}

	for i := range sum {
		sum[i] /= float64(len(vectors))
	}
	return sum
}

func saveMatrix(filename string, m mat.Matrix) error {
	// Создаем файл
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	// Получаем размеры матрицы
	r, c := m.Dims()

	// Записываем матрицу построчно
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			// Форматируем число с 4 знаками после запятой
			if j > 0 {
				f.WriteString("\t") // Используем табуляцию как разделитель
			}
			fmt.Fprintf(f, "%.4f", m.At(i, j))
		}
		f.WriteString("\n") // Переход на новую строку
	}

	return nil
}
