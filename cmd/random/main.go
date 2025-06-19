package main

import (
	"classifier-go/pkg/randomnormal"

	"fmt"
	"strings"
)

func main() {
	// Инициализация генератора с параметрами:
	// mean (среднее) = 0.5
	// stddev (стандартное отклонение) = 0.2
	// min = 0.0
	// max = 1.0
	gen := randomnormal.NewNormalRandGenerator(3e-3, (5e-3-5e-4)/6, 5e-4, 5e-3)

	// Генерация 10 чисел
	numbers := gen.RandN(10)
	fmt.Println("Сгенерированные числа:")
	for i, num := range numbers {
		fmt.Printf("%d: %.4f\n", i+1, num)
	}

	// Пример гистограммы (для визуализации распределения)
	fmt.Println("\nГистограмма распределения (20 интервалов):")
	hist := make([]int, 20)
	for _, num := range gen.RandN(10000) {
		bin := int((num - gen.Min()) / (gen.Max() - gen.Min()) * 20)
		if bin >= 0 && bin < 20 {
			hist[bin]++
		}
	}

	maxCount := 0
	for _, count := range hist {
		if count > maxCount {
			maxCount = count
		}
	}

	for i, count := range hist {
		bar := strings.Repeat("█", int(float64(count)/float64(maxCount)*50))
		fmt.Printf("%.2f-%.2f: %s %d\n",
			float64(i)/20, float64(i+1)/20, bar, count)
	}
}
