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
	"slices"
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

	// Если нужно сглаживание, то применяем фильтр Гаусса
	if cfg.DoSmooth {
		gs := convolve.NewGaussianKernel(float64(cfg.SigmaT), float64(cfg.SigmaH), cfg.Size)
		r := depolarizationMatrix.Slice(0, rows, 1, cols).(*mat.Dense)
		output := gs.Convolve(r)
		r.Copy(output)

		r = fluorescenceCapacityMatrix.Slice(0, rows, 1, cols).(*mat.Dense)
		output = gs.Convolve(r)
		r.Copy(output)
	}

	// переводим реперные значения деполяризации для исключения появления ошибки деления на ноль
	delta_d := cfg.DeltaDust / (1.0 + cfg.DeltaDust)
	delta_u := cfg.DeltaUrban / (1.0 + cfg.DeltaUrban)
	delta_s := cfg.DeltaSoot / (1.0 + cfg.DeltaSoot)

	// Generate random values
	gfuMin := cfg.GfUrban * (1.0 - cfg.VariationCoefficient)
	gfuMax := cfg.GfUrban * (1.0 + cfg.VariationCoefficient)
	genGfUrban := randomnormal.NewNormalRandGenerator(cfg.GfUrban, (gfuMax-gfuMin)/4, gfuMin, gfuMax)
	gfUs := genGfUrban.RandN(cfg.NumPoints)

	gfsMin := cfg.GfSoot * (1.0 - cfg.VariationCoefficient)
	gfsMax := cfg.GfSoot * (1.0 + cfg.VariationCoefficient)
	genGfSoot := randomnormal.NewNormalRandGenerator(cfg.GfSoot, (gfsMax-gfsMin)/4, gfsMin, gfsMax)
	gfSs := genGfSoot.RandN(cfg.NumPoints)

	gfdMin := cfg.GfDust * (1.0 - cfg.VariationCoefficient)
	gfdMax := cfg.GfDust * (1.0 + cfg.VariationCoefficient)
	genGfDust := randomnormal.NewNormalRandGenerator(cfg.GfDust, (gfdMax-gfdMin)/4, gfdMin, gfdMax)
	gfDs := genGfDust.RandN(cfg.NumPoints)

	delta_s_min := delta_s * (1.0 - cfg.VariationCoefficient)
	delta_s_max := delta_s * (1.0 + cfg.VariationCoefficient)
	genDeltaSoot := randomnormal.NewNormalRandGenerator(delta_s, (delta_s_max-delta_s_min)/4, delta_s_min, delta_s_max)
	delta_ss := genDeltaSoot.RandN(cfg.NumPoints)

	delta_u_min := delta_u * (1.0 - cfg.VariationCoefficient)
	delta_u_max := delta_u * (1.0 + cfg.VariationCoefficient)
	genDeltaUrban := randomnormal.NewNormalRandGenerator(delta_u, (delta_u_max-delta_u_min)/4, delta_u_min, delta_u_max)
	delta_us := genDeltaUrban.RandN(cfg.NumPoints)

	delta_d_min := delta_d * (1.0 - cfg.VariationCoefficient)
	delta_d_max := delta_d * (1.0 + cfg.VariationCoefficient)
	genDeltaDust := randomnormal.NewNormalRandGenerator(delta_d, (delta_d_max-delta_d_min)/4, delta_d_min, delta_d_max)
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

	Gf_u_n := mat.NewDense(r, c, nil)
	Gf_d_n := mat.NewDense(r, c, nil)
	Gf_s_n := mat.NewDense(r, c, nil)
	Delta_u_n := mat.NewDense(r, c, nil)
	Delta_d_n := mat.NewDense(r, c, nil)
	Delta_s_n := mat.NewDense(r, c, nil)

	// Copy first column (time/altitude)
	for i := range r {
		Eta_u.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Eta_s.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Eta_d.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Gf_u_n.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Gf_d_n.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Gf_s_n.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Delta_u_n.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Delta_d_n.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Delta_s_n.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
	}

	// Создаем пул воркеров
	numWorkers := runtime.NumCPU() // Используем все доступные ядра
	taskQueue := make(chan task, cfg.NumPoints)
	resultQueue := make(chan result, cfg.NumPoints)

	// Запускаем воркеры
	var wg sync.WaitGroup
	for range numWorkers {
		wg.Add(1)
		go worker(taskQueue, resultQueue, &wg)
	}

	newGf := make([]float64, 3)
	newDelta := make([]float64, 3)
	// Main processing loop
	for i := 0; i < r; i++ {
		fmt.Printf("Row = %d\n", i)
		for j := 1; j < c; j++ {
			delta_meas := depolarizationMatrix.At(i, j) // 100.0
			GF_meas := fluorescenceCapacityMatrix.At(i, j)

			// Отправляем задачи воркерам
			for k := range cfg.NumPoints {
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
			tmp_eta := make([]result, 0, cfg.NumPoints)
			for k := 0; k < cfg.NumPoints; k++ {
				res := <-resultQueue
				if res.Valid { // nil означает недопустимое решение
					tmp_eta = append(tmp_eta, res)
				}
			}

			if len(tmp_eta) == 0 {
				fmt.Printf("Warning: no valid solutions for point (%d,%d)\n", i, j)
				continue
			}

			// Average the valid estimates
			etas_mean := averageVectors(tmp_eta, 0.1)
			Eta_u.Set(i, j, etas_mean[0].X)
			Eta_d.Set(i, j, etas_mean[1].X)
			Eta_s.Set(i, j, etas_mean[2].X)

			Delta_u_n.Set(i, j, etas_mean[0].Delta)
			Delta_d_n.Set(i, j, etas_mean[1].Delta)
			Delta_s_n.Set(i, j, etas_mean[2].Delta)
			Gf_u_n.Set(i, j, etas_mean[0].Gf)
			Gf_d_n.Set(i, j, etas_mean[1].Gf)
			Gf_s_n.Set(i, j, etas_mean[2].Gf)

			newGf[0] += etas_mean[0].Gf
			newGf[1] += etas_mean[1].Gf
			newGf[2] += etas_mean[2].Gf

			newDelta[0] += (etas_mean[0].Delta / (1 - etas_mean[0].Delta))
			newDelta[1] += (etas_mean[1].Delta / (1 - etas_mean[1].Delta))
			newDelta[2] += (etas_mean[2].Delta / (1 - etas_mean[2].Delta))
		}
	}

	nAvg := r * (c - 1)
	for i := range 3 {
		newGf[i] /= float64(nAvg)
		newDelta[i] /= float64(nAvg)
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

	err = saveMatrix(cfg.InputDir+"Gf_u.csv", Gf_u_n)
	if err != nil {
		log.Fatal("Error saving Gf_u_n matrix", err)
	}

	err = saveMatrix(cfg.InputDir+"Gf_d.csv", Gf_d_n)
	if err != nil {
		log.Fatal("Error saving Gf_d_n matrix", err)
	}

	err = saveMatrix(cfg.InputDir+"Gf_s.csv", Gf_s_n)
	if err != nil {
		log.Fatal("Error saving Gf_s_n matrix", err)
	}

	err = saveMatrix(cfg.InputDir+"Delta_u.csv", Delta_u_n)
	if err != nil {
		log.Fatal("Error saving Delta_u matrix", err)
	}

	err = saveMatrix(cfg.InputDir+"Delta_d.csv", Delta_d_n)
	if err != nil {
		log.Fatal("Error saving Delta_d matrix", err)
	}

	err = saveMatrix(cfg.InputDir+"Delta_s.csv", Delta_s_n)
	if err != nil {
		log.Fatal("Error saving Delta_s matrix", err)
	}

	err = saveMatrix(cfg.InputDir+"Gf_s.csv", Gf_s_n)
	if err != nil {
		log.Fatal("Error saving Gf_s_n matrix", err)
	}
	heatmapplotter.MakeHeatmapPlot(depolarizationMatrix, "Dep", cfg.InputDir+"Dep.pdf")
	heatmapplotter.MakeHeatmapPlot(Eta_d, "Eta_d", cfg.InputDir+"Eta_d.pdf")
	heatmapplotter.MakeHeatmapPlot(Eta_u, "Eta_u", cfg.InputDir+"Eta_u.pdf")
	heatmapplotter.MakeHeatmapPlot(Eta_s, "Eta_s", cfg.InputDir+"Eta_s.pdf")

	fmt.Println("Tuned parameters:")
	fmt.Printf("Gf_u: %.3e, delta_u: %.3f\n", newGf[0], newDelta[0])
	fmt.Printf("Gf_d: %.3e, delta_d: %.3f\n", newGf[1], newDelta[1])
	fmt.Printf("Gf_s: %.3e, delta_s: %.3f\n", newGf[2], newDelta[2])
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

// Структура для хранения результатов
type result struct {
	X     []float64
	F     float64
	Gf    [3]float64 //Gfd, Gfu, Gfs
	Delta [3]float64 //Deltad. Deltau, Deltas
	Valid bool
}

type avgresult struct {
	X     float64
	Delta float64
	Gf    float64
}

// Функция воркера
func worker(tasks <-chan task, results chan<- result, wg *sync.WaitGroup) {
	defer wg.Done()
	for t := range tasks {
		ntas_i, F_i, err := classifySinglePoint(t.GF_meas, t.delta_meas/(1+t.delta_meas),
			t.GF_u_k, t.GF_d_k, t.GF_s_k, t.delta_u_k, t.delta_d_k, t.delta_s_k)

		if err == nil && isValidSolution(ntas_i) {
			results <- result{
				X:     ntas_i,
				F:     F_i,
				Gf:    [3]float64{t.GF_u_k, t.GF_d_k, t.GF_s_k},
				Delta: [3]float64{t.delta_u_k, t.delta_d_k, t.delta_s_k},
				Valid: true,
			}
		} else {
			results <- result{Valid: false}
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

func classifySinglePoint(GF_meas, delta_meas, GF_u, GF_d, GF_s, delta_u, delta_d, delta_s float64) ([]float64, float64, error) {
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
		return nil, 0, fmt.Errorf("optimization failed: %v", err)
	}

	// Дополнительная проверка границ
	for i := range result.X {
		result.X[i] = math.Max(0, math.Min(1, result.X[i]))
	}

	// Проверка суммы параметров (должна быть близка к 1)
	sum := floats.Sum(result.X)
	if math.Abs(sum-1) > 0.01 { // Допустимое отклонение 1%
		return nil, 0, fmt.Errorf("invalid solution: sum of parameters = %f", sum)
	}

	return result.X, result.F, nil
}

func averageVectors(vectors []result, avgFrac float64) []avgresult {
	if len(vectors) == 0 {
		return []avgresult{}
	}

	// filtered_vectors := Filter(vectors, func(v result) bool {
	// 	return v.Valid
	// })

	slices.SortFunc(vectors, func(a result, b result) int {
		if a.F < b.F {
			return -1
		} else if a.F > b.F {
			return 1
		} else {
			return 0
		}
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
	//println(vectors[0].F, vectors[Ntot-1].F)
	for i := range sum {
		sum[i].Delta /= float64(Ntot)
		sum[i].Gf /= float64(Ntot)
		sum[i].X /= float64(Ntot)
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
			fmt.Fprintf(f, "%.4e", m.At(i, j))
		}
		f.WriteString("\n") // Переход на новую строку
	}

	return nil
}

// Predicate — функция-предикат, определяющая условие фильтрации
type Predicate[T any] func(T) bool

// Filter — фильтрует срез, оставляя только элементы, удовлетворяющие предикату
func Filter[T any](src []T, pred Predicate[T]) []T {
	var filtered []T
	for _, item := range src {
		if pred(item) {
			filtered = append(filtered, item)
		}
	}
	return filtered
}
