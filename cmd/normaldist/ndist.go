package main

import (
	"classifier-go/pkg/normalboxmueller"
	"fmt"
	"sort"
)

// ParticleData представляет данные о частицах
type ParticleData struct {
	Radii []float64 // Радиусы частиц
}

// Histogram представляет гистограмму распределения
type Histogram struct {
	Bins   []float64 // Границы бинов
	Counts []int     // Количество частиц в каждом бине
}

// NewHistogram создает новую гистограмму с заданными границами бинов
func NewHistogram(bins []float64) *Histogram {
	return &Histogram{
		Bins:   bins,
		Counts: make([]int, len(bins)-1),
	}
}

// CalculateHistogram вычисляет гистограмму распределения частиц по размерам
func (pd *ParticleData) CalculateHistogram(bins []float64) *Histogram {
	hist := NewHistogram(bins)

	// Сортируем радиусы для более эффективного подсчета
	sort.Float64s(pd.Radii)

	// Инициализируем индекс текущего бина
	binIndex := 0

	for _, radius := range pd.Radii {
		// Находим подходящий бин для текущего радиуса
		for radius >= hist.Bins[binIndex+1] && binIndex < len(hist.Counts)-1 {
			binIndex++
		}

		// Если радиус попадает в диапазон гистограммы
		if radius >= hist.Bins[binIndex] && radius < hist.Bins[binIndex+1] {
			hist.Counts[binIndex]++
		}
		// Радиусы вне диапазона игнорируем
	}

	return hist
}

// Print выводит гистограмму в читаемом формате
func (h *Histogram) Print() {
	for i := 0; i < len(h.Counts); i++ {
		lower := h.Bins[i]
		upper := h.Bins[i+1]
		fmt.Printf("[%.2f - %.2f): %d\n", lower, upper, h.Counts[i])
	}
}

func main() {
	pp := normalboxmueller.NewNormalDistParams(0.25, 0.1/6, 0.2, 0.3)
	radii := make([]float64, 1000)
	pp.GenerateVector(radii)
	// Пример данных - радиусы частиц в микронах
	particleData := &ParticleData{
		Radii: radii,
	}

	// Создаем бины для гистограммы (можно адаптировать под ваши нужды)
	db := (pp.High - pp.Low) / 10
	bins := make([]float64, 11)
	for i := range bins {
		bins[i] = pp.Low + db*float64(i)
	}

	// Вычисляем гистограмму
	hist := particleData.CalculateHistogram(bins)

	// Выводим результаты
	fmt.Println("Гистограмма распределения частиц по размерам:")
	hist.Print()

	// Дополнительная статистика
	total := 0
	for _, count := range hist.Counts {
		total += count
	}
	fmt.Printf("\nВсего частиц: %d\n", total)
}
