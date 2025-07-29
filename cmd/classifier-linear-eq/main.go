/*
Copyright © 2025 Konstantin Shmirko <physicist2018@vk.com>
*/
package main

import (
	"classifier-go/internal/config"
	"classifier-go/internal/presenter"
	prepare "classifier-go/internal/solver"
	"classifier-go/pkg/convolve"
	"classifier-go/pkg/readmatrix"
	"fmt"
	"log"
	"path/filepath"
	"runtime"
	"sync"

	"gonum.org/v1/gonum/mat"
)

func main() {
	cfg := config.Parse()
	log.Println("Starting classifier...")

	// Load input matrices
	flMatrix, err := readmatrix.ReadMatrix(filepath.Join(cfg.InputDir, "FL_cap.txt"))
	if err != nil {
		log.Fatalf("Ошибка чтения матрицы флуоресценции: %v\n", err)
	}

	depMatrix, err := readmatrix.ReadMatrix(filepath.Join(cfg.InputDir, "Dep.txt"))
	if err != nil {
		log.Fatalf("Ошибка чтения матрицы деполяризации: %v\n", err)
	}

	if cfg.DoSmooth {
		rows, cols := depMatrix.Dims()
		gs := convolve.NewGaussianKernel(float64(cfg.SigmaT), float64(cfg.SigmaH), cfg.Size)
		smoothMatrix(flMatrix, gs, rows, cols)
		smoothMatrix(depMatrix, gs, rows, cols)
	}

	Kernels := prepare.PrepareMatricesA(cfg)
	r, c := flMatrix.Dims()

	// Создаем матрицы для результатов
	eta_u := mat.NewDense(r, c, nil)
	eta_s := mat.NewDense(r, c, nil)
	eta_d := mat.NewDense(r, c, nil)
	delta_u := mat.NewDense(r, c, nil)
	delta_s := mat.NewDense(r, c, nil)
	delta_d := mat.NewDense(r, c, nil)
	gf_u := mat.NewDense(r, c, nil)
	gf_s := mat.NewDense(r, c, nil)
	gf_d := mat.NewDense(r, c, nil)
	errr := mat.NewDense(r, c, nil)

	navg := int(float64(cfg.AvgPercent) * float64(cfg.NumPoints))

	// Устанавливаем количество рабочих goroutines
	numWorkers := runtime.NumCPU() // Можно настроить в зависимости от количества CPU
	var wg sync.WaitGroup
	workChan := make(chan int, r)

	// Запускаем workers
	for _ = range numWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range workChan {
				fmt.Printf("Start procesing %d row\n", i)
				// Обработка первого столбца
				errr.Set(i, 0, flMatrix.At(i, 0))
				eta_d.Set(i, 0, flMatrix.At(i, 0))
				eta_u.Set(i, 0, flMatrix.At(i, 0))
				eta_s.Set(i, 0, flMatrix.At(i, 0))
				delta_d.Set(i, 0, flMatrix.At(i, 0))
				delta_u.Set(i, 0, flMatrix.At(i, 0))
				delta_s.Set(i, 0, flMatrix.At(i, 0))
				gf_d.Set(i, 0, flMatrix.At(i, 0))
				gf_u.Set(i, 0, flMatrix.At(i, 0))
				gf_s.Set(i, 0, flMatrix.At(i, 0))

				// Обработка остальных столбцов
				for j := 1; j < c; j++ {
					depMatrix.Set(i, j, depMatrix.At(i, j)/100.0)
					b := prepare.FormBVector(depMatrix.At(i, j), flMatrix.At(i, j))
					solutions := mat.NewDense(cfg.NumPoints, 10, nil)

					for k := range cfg.NumPoints {
						tmpA := Kernels[k]

						x, F := prepare.SolveWithBounds(tmpA, b, []float64{0, 0, 0}, []float64{1, 1, 1})

						solutions.Set(k, 0, F)
						solutions.Set(k, 1, x.AtVec(0))
						solutions.Set(k, 2, x.AtVec(1))
						solutions.Set(k, 3, x.AtVec(2))
						solutions.Set(k, 4, tmpA.At(0, 0))
						solutions.Set(k, 5, tmpA.At(0, 1))
						solutions.Set(k, 6, tmpA.At(0, 2))
						solutions.Set(k, 7, tmpA.At(1, 0))
						solutions.Set(k, 8, tmpA.At(1, 1))
						solutions.Set(k, 9, tmpA.At(1, 2))
					}

					prepare.SortRows(solutions, 0)
					finalResult := prepare.AveragePRows(solutions, navg)
					errr.Set(i, j, finalResult.AtVec(0))
					eta_d.Set(i, j, finalResult.AtVec(1))
					eta_u.Set(i, j, finalResult.AtVec(2))
					eta_s.Set(i, j, finalResult.AtVec(3))
					delta_d.Set(i, j, finalResult.AtVec(4))
					delta_u.Set(i, j, finalResult.AtVec(5))
					delta_s.Set(i, j, finalResult.AtVec(6))
					gf_d.Set(i, j, finalResult.AtVec(7))
					gf_u.Set(i, j, finalResult.AtVec(8))
					gf_s.Set(i, j, finalResult.AtVec(9))
				}
				fmt.Printf("Stop procesing %d row\n", i)
			}
		}()
	}

	// Отправляем задачи в канал
	for i := range r {
		workChan <- i
	}
	close(workChan)

	// Ожидаем завершения всех workers
	wg.Wait()

	// Генерация результатов
	presenter.GenerateHeatmap(filepath.Join(cfg.InputDir, "Dep.pdf"), "Depolarization", depMatrix)
	presenter.GenerateHeatmap(filepath.Join(cfg.InputDir, "Eta_u.pdf"), "Urban aerosol fraction", eta_u)
	presenter.GenerateHeatmap(filepath.Join(cfg.InputDir, "Eta_d.pdf"), "Dust aerosol fraction", eta_d)
	presenter.GenerateHeatmap(filepath.Join(cfg.InputDir, "Eta_s.pdf"), "Smoke aerosol fraction", eta_s)

	presenter.SaveDenseToCSV(eta_u, filepath.Join(cfg.InputDir, "Eta_u.csv"))
	presenter.SaveDenseToCSV(eta_d, filepath.Join(cfg.InputDir, "Eta_d.csv"))
	presenter.SaveDenseToCSV(eta_s, filepath.Join(cfg.InputDir, "Eta_s.csv"))
	presenter.SaveDenseToCSV(delta_u, filepath.Join(cfg.InputDir, "delta_u.csv"))
	presenter.SaveDenseToCSV(delta_d, filepath.Join(cfg.InputDir, "delta_d.csv"))
	presenter.SaveDenseToCSV(delta_s, filepath.Join(cfg.InputDir, "delta_s.csv"))
	presenter.SaveDenseToCSV(gf_u, filepath.Join(cfg.InputDir, "gf_u.csv"))
	presenter.SaveDenseToCSV(gf_d, filepath.Join(cfg.InputDir, "gf_d.csv"))
	presenter.SaveDenseToCSV(gf_s, filepath.Join(cfg.InputDir, "gf_s.csv"))
	presenter.SaveDenseToCSV(errr, filepath.Join(cfg.InputDir, "errr.csv"))

	fmt.Println("All results saved.")
	fmt.Printf("Tuned values of:\n")
	fmt.Printf("Gf_d   : %.3e\n", avgMatrix(gf_d))
	fmt.Printf("Gf_u   : %.3e\n", avgMatrix(gf_u))
	fmt.Printf("Gf_s   : %.3e\n", avgMatrix(gf_s))
	fmt.Printf("delta_d: %.3f\n", avgMatrix(delta_d))
	fmt.Printf("delta_u: %.3f\n", avgMatrix(delta_u))
	fmt.Printf("delta_s: %.3f\n", avgMatrix(delta_s))
}

func smoothMatrix(matrix *mat.Dense, gs *convolve.ConvolveKernel, rows, cols int) {
	r := matrix.Slice(0, rows, 1, cols).(*mat.Dense)
	output := gs.Convolve(r)
	r.Copy(output)
}

func avgMatrix(m *mat.Dense) float64 {
	r, c := m.Dims()
	var sum float64
	for i := range r {
		for j := 1; j < c; j++ {
			sum += m.At(i, j)
		}
	}
	return sum / float64(r*(c-1))
}

// /*
// Copyright © 2025 Kpnstantin Shmirko <physicist2018@vk.com>
// */
// package main

// import (
// 	"classifier-go/internal/config"
// 	"classifier-go/internal/presenter"
// 	prepare "classifier-go/internal/solver"
// 	"classifier-go/pkg/readmatrix"
// 	"log"
// 	"path/filepath"

// 	"gonum.org/v1/gonum/mat"
// )

// func main() {
// 	cfg := config.Parse()
// 	log.Println("Starting classifier...")

// 	// Load input matrices
// 	flMatrix, err := readmatrix.ReadMatrix(filepath.Join(cfg.InputDir, "FL_cap.txt"))
// 	if err != nil {
// 		log.Fatalf("Ошибка чтения матрицы флуоресценции: %v\n", err)
// 	}

// 	depMatrix, err := readmatrix.ReadMatrix(filepath.Join(cfg.InputDir, "Dep.txt"))
// 	if err != nil {
// 		log.Fatalf("Ошибка чтения матрицы деполяризации: %v\n", err)
// 	}

// 	Kernels := prepare.PrepareMatricesA(cfg)
// 	r, c := flMatrix.Dims()

// 	eta_u := mat.NewDense(r, c, nil)
// 	eta_s := mat.NewDense(r, c, nil)
// 	eta_d := mat.NewDense(r, c, nil)
// 	delta_u := mat.NewDense(r, c, nil)
// 	delta_s := mat.NewDense(r, c, nil)
// 	delta_d := mat.NewDense(r, c, nil)
// 	gf_u := mat.NewDense(r, c, nil)
// 	gf_s := mat.NewDense(r, c, nil)
// 	gf_d := mat.NewDense(r, c, nil)
// 	errr := mat.NewDense(r, c, nil)

// 	navg := int(float64(cfg.AvgPercent) * float64(cfg.NumPoints))
// 	for i := 0; i < r; i++ {
// 		errr.Set(i, 0, flMatrix.At(i, 0))
// 		eta_d.Set(i, 0, flMatrix.At(i, 0))
// 		eta_u.Set(i, 0, flMatrix.At(i, 0))
// 		eta_s.Set(i, 0, flMatrix.At(i, 0))
// 		delta_d.Set(i, 0, flMatrix.At(i, 0))
// 		delta_u.Set(i, 0, flMatrix.At(i, 0))
// 		delta_s.Set(i, 0, flMatrix.At(i, 0))
// 		gf_d.Set(i, 0, flMatrix.At(i, 0))
// 		gf_u.Set(i, 0, flMatrix.At(i, 0))
// 		gf_s.Set(i, 0, flMatrix.At(i, 0))
// 		for j := 1; j < c; j++ {
// 			depMatrix.Set(i, j, depMatrix.At(i, j)/100.0)
// 			b := prepare.FormBVector(depMatrix.At(i, j), flMatrix.At(i, j))
// 			solutions := mat.NewDense(cfg.NumPoints, 10, nil)
// 			for k := range cfg.NumPoints {

// 				tmpA := Kernels[k]

// 				x, F := prepare.SolveWithBounds(tmpA, b, []float64{0, 0, 0}, []float64{1, 1, 1})

// 				solutions.Set(k, 0, F)
// 				solutions.Set(k, 1, x.AtVec(0))
// 				solutions.Set(k, 2, x.AtVec(1))
// 				solutions.Set(k, 3, x.AtVec(2))
// 				solutions.Set(k, 4, tmpA.At(0, 0))
// 				solutions.Set(k, 5, tmpA.At(0, 1))
// 				solutions.Set(k, 6, tmpA.At(0, 2))
// 				solutions.Set(k, 7, tmpA.At(1, 0))
// 				solutions.Set(k, 8, tmpA.At(1, 1))
// 				solutions.Set(k, 9, tmpA.At(1, 2))
// 			}

// 			prepare.SortRows(solutions, 0)
// 			finalResult := prepare.AveragePRows(solutions, navg)
// 			errr.Set(i, j, finalResult.AtVec(0))
// 			eta_d.Set(i, j, finalResult.AtVec(1))
// 			eta_u.Set(i, j, finalResult.AtVec(2))
// 			eta_s.Set(i, j, finalResult.AtVec(3))
// 			delta_d.Set(i, j, finalResult.AtVec(4))
// 			delta_u.Set(i, j, finalResult.AtVec(5))
// 			delta_s.Set(i, j, finalResult.AtVec(6))
// 			gf_d.Set(i, j, finalResult.AtVec(7))
// 			gf_u.Set(i, j, finalResult.AtVec(8))
// 			gf_s.Set(i, j, finalResult.AtVec(9))

// 		}
// 	}
// 	presenter.GenerateHeatmap(filepath.Join(cfg.InputDir, "Dep.pdf"), "Depolarization", depMatrix)
// 	presenter.GenerateHeatmap(filepath.Join(cfg.InputDir, "Eta_u.pdf"), "Urban aerosol fraction", eta_u)
// 	presenter.GenerateHeatmap(filepath.Join(cfg.InputDir, "Eta_d.pdf"), "Dust aerosol fraction", eta_d)
// 	presenter.GenerateHeatmap(filepath.Join(cfg.InputDir, "Eta_s.pdf"), "Smoke aerosol fraction", eta_s)

// 	presenter.SaveDenseToCSV(eta_u, filepath.Join(cfg.InputDir, "Eta_u.csv"))
// 	presenter.SaveDenseToCSV(eta_d, filepath.Join(cfg.InputDir, "Eta_d.csv"))
// 	presenter.SaveDenseToCSV(eta_s, filepath.Join(cfg.InputDir, "Eta_s.csv"))
// 	presenter.SaveDenseToCSV(delta_u, filepath.Join(cfg.InputDir, "delta_u.csv"))
// 	presenter.SaveDenseToCSV(delta_d, filepath.Join(cfg.InputDir, "delta_d.csv"))
// 	presenter.SaveDenseToCSV(delta_s, filepath.Join(cfg.InputDir, "delta_s.csv"))
// 	presenter.SaveDenseToCSV(gf_u, filepath.Join(cfg.InputDir, "gf_u.csv"))
// 	presenter.SaveDenseToCSV(gf_d, filepath.Join(cfg.InputDir, "gf_d.csv"))
// 	presenter.SaveDenseToCSV(gf_s, filepath.Join(cfg.InputDir, "gf_s.csv"))
// 	presenter.SaveDenseToCSV(errr, filepath.Join(cfg.InputDir, "errr.csv"))

// }
