package convolve

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type ConvolveKernel struct {
	kernel *mat.Dense
}

func NewGaussianKernel(sigmaT, sigmaH float64, size int) *ConvolveKernel {
	kernel := mat.NewDense(size, size, nil)
	sum := 0.0

	// Центр ядра
	center := float64(size-1) / 2.0

	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			t := float64(i) - center
			h := float64(j) - center
			value := math.Exp(-((t*t)/(sigmaT*sigmaT) + (h*h)/(sigmaH*sigmaH)))
			kernel.Set(i, j, value)
			sum += value
		}
	}

	// Нормализация ядра
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			v := kernel.At(i, j)
			kernel.Set(i, j, v/sum)
		}
	}

	return &ConvolveKernel{kernel}
}

// Функция для добавления Zero-Padding к матрице
func padMatrix(input *mat.Dense, padding int) *mat.Dense {
	rows, cols := input.Dims()
	totalRows := rows + 2*padding
	totalCols := cols + 2*padding

	padded := mat.NewDense(totalRows, totalCols, nil)

	// Заполняем середину оригинальной матрицей
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			padded.Set(i+padding, j+padding, input.At(i, j))
		}
	}

	return padded
}

func (ck *ConvolveKernel) Convolve(input *mat.Dense) *mat.Dense {
	inputRows, inputCols := input.Dims()
	kernelRows, kernelCols := ck.kernel.Dims()

	// Применяем zero padding
	padding := kernelRows / 2
	paddedInput := padMatrix(input, padding)

	// Новую матрицу делаем того же размера, что и исходная
	output := mat.NewDense(inputRows, inputCols, nil)

	for i := 0; i < inputRows; i++ {
		for j := 0; j < inputCols; j++ {
			sum := 0.0
			for ki := 0; ki < kernelRows; ki++ {
				for kj := 0; kj < kernelCols; kj++ {
					inputRow := i + ki
					inputCol := j + kj
					sum += paddedInput.At(inputRow, inputCol) * ck.kernel.At(ki, kj)
				}
			}
			output.Set(i, j, sum)
		}
	}

	return output
}
