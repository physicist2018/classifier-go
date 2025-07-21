package main

import (
	"classifier-go/pkg/convolve"
	"classifier-go/pkg/heatmapplotter"
	"classifier-go/pkg/readmatrix"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
)

type GaussClass struct {
	Title     string        `json:"title"`
	Center    [2]float64    `json:"center"`
	CovMatrix [2][2]float64 `json:"cov_matrix"`
}

// Calculate the determinant of a 2x2 matrix
func determinant(matrix [2][2]float64) float64 {
	return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
}

// Calculate the inverse of a 2x2 matrix
func inverse(matrix [2][2]float64) [2][2]float64 {
	det := determinant(matrix)
	if det == 0 {
		panic("Matrix is not invertible")
	}
	return [2][2]float64{
		{matrix[1][1] / det, -matrix[0][1] / det},
		{-matrix[1][0] / det, matrix[0][0] / det},
	}
}

// Calculate gaussian probability density function
func gaussianProbability(point [2]float64, class GaussClass) float64 {
	deltaX := point[0] - class.Center[0]
	deltaY := point[1] - class.Center[1]

	// Вычисляем детерминант и обратную матрицу ковариации
	det := determinant(class.CovMatrix)
	if det == 0 {
		return 0
	}

	invCov := inverse(class.CovMatrix)

	exponent := -0.5 * (deltaX*invCov[0][0]*deltaX + deltaX*invCov[0][1]*deltaY + deltaY*invCov[1][0]*deltaX + deltaY*invCov[1][1]*deltaY)

	return (1 / (2 * math.Pi * math.Sqrt(det))) * math.Exp(exponent)
}

// Функция для чтения классов из файла JSON
func readClassesFromJSON(filename string) ([]GaussClass, error) {
	var classes []GaussClass
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	err = json.Unmarshal(data, &classes)
	if err != nil {
		return nil, err
	}

	return classes, nil
}

// Функция для определения нормированных вероятностей принадлежности точки к классам
func classifyPoint(point [2]float64, classes []GaussClass) []float64 {
	probabilities := make([]float64, len(classes))
	sumProb := 0.0

	// Вычисляем вероятности
	for i, class := range classes {
		probabilities[i] = gaussianProbability(point, class)
		sumProb += probabilities[i]
	}

	// Нормируем вероятности
	if sumProb > 0 {
		for i := range probabilities {
			probabilities[i] /= sumProb
		}
	}

	return probabilities
}

func main() {
	inputDir := flag.String("input-dir", "./", "input directory")
	outputDir := flag.String("output-dir", "./", "output directory")
	doSmooth := flag.Bool("do-smooth", true, "activating smoothing")
	sigmaH := flag.Int("sigma-h", 5, "spatial smoothing size in bins")
	sigmaT := flag.Int("sigma-t", 3, "temporal smoothing size in bins")
	smoothSize := flag.Int("size", 7, "kernel size in bins")

	flag.Parse()
	println(*inputDir + "FL_Cap.txt")
	fluorescenceCapacityMatrix, err := readmatrix.ReadMatrix(*inputDir + "FL_Cap.txt")
	if err != nil {
		log.Fatal("Error reading Fluorescence Capacity Matrix", err)
	}
	depolarizationMatrix, err := readmatrix.ReadMatrix(*inputDir + "Dep.txt")
	if err != nil {
		log.Fatal("Error reading depolarization Matrix", err)
	}
	rows, cols := depolarizationMatrix.Dims()
	tmpSlice := depolarizationMatrix.Slice(0, rows, 1, cols).(*mat.Dense)
	tmpSlice.Scale(0.01, tmpSlice)

	// Если нужно сглаживание, то применяем фильтр Гаусса
	if *doSmooth {
		gs := convolve.NewGaussianKernel(float64(*sigmaT), float64(*sigmaH), *smoothSize)
		r := depolarizationMatrix.Slice(0, rows, 1, cols).(*mat.Dense)
		output := gs.Convolve(r)
		r.Copy(output)

		r = fluorescenceCapacityMatrix.Slice(0, rows, 1, cols).(*mat.Dense)
		output = gs.Convolve(r)
		r.Copy(output)
	}

	// Читаем классы из файла JSON
	classes, err := readClassesFromJSON("classes.json")
	if err != nil {
		fmt.Println(err)
		return
	}

	// Initialize result matrices
	Eta_u := mat.NewDense(rows, cols, nil)
	Eta_s := mat.NewDense(rows, cols, nil)
	Eta_d := mat.NewDense(rows, cols, nil)
	Eta_p := mat.NewDense(rows, cols, nil)
	Eta_w := mat.NewDense(rows, cols, nil)

	// Copy first column (time/altitude)
	for i := range rows {
		Eta_u.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Eta_s.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Eta_d.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Eta_p.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
		Eta_w.Set(i, 0, fluorescenceCapacityMatrix.At(i, 0))
	}

	for r := 0; r < rows; r++ {
		for c := 1; c < cols; c++ {
			point := [2]float64{depolarizationMatrix.At(r, c), fluorescenceCapacityMatrix.At(r, c)}
			probs := classifyPoint(point, classes)
			Eta_d.Set(r, c, probs[0])
			Eta_s.Set(r, c, probs[1])
			Eta_u.Set(r, c, probs[2])
			Eta_p.Set(r, c, probs[3])
			Eta_w.Set(r, c, probs[4])
		}
	}

	err = saveMatrix(*outputDir+"Eta_s.csv", Eta_s)
	if err != nil {
		log.Fatal("Error saving Eta_s matrix", err)
	}

	err = saveMatrix(*outputDir+"Eta_d.csv", Eta_d)
	if err != nil {
		log.Fatal("Error saving Eta_d matrix", err)
	}

	err = saveMatrix(*outputDir+"Eta_u.csv", Eta_u)
	if err != nil {
		log.Fatal("Error saving Eta_u matrix", err)
	}

	err = saveMatrix(*outputDir+"Eta_p.csv", Eta_p)
	if err != nil {
		log.Fatal("Error saving Eta_p matrix", err)
	}

	err = saveMatrix(*outputDir+"Eta_w.csv", Eta_w)
	if err != nil {
		log.Fatal("Error saving Eta_w matrix", err)
	}

	heatmapplotter.MakeHeatmapPlot(depolarizationMatrix, "Dep", *outputDir+"Dep.pdf")
	heatmapplotter.MakeHeatmapPlot(Eta_d, "Eta_d", *outputDir+"Eta_d.pdf")
	heatmapplotter.MakeHeatmapPlot(Eta_u, "Eta_u", *outputDir+"Eta_u.pdf")
	heatmapplotter.MakeHeatmapPlot(Eta_s, "Eta_s", *outputDir+"Eta_s.pdf")
	heatmapplotter.MakeHeatmapPlot(Eta_p, "Eta_p", *outputDir+"Eta_p.pdf")
	heatmapplotter.MakeHeatmapPlot(Eta_w, "Eta_w", *outputDir+"Eta_w.pdf")
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
