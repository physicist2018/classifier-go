package normalboxmueller

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// NormalDistParams represents the parameters for a normal distribution.
// It contains the mean, standard deviation, lower bound, and upper bound of the distribution.
type NormalDistParams struct {
	Mean   float64 `json:"mean"`
	StdDev float64 `json:"std_dev"`
	Low    float64 `json:"low"`
	High   float64 `json:"high"`
}

// NewNormalDistParams creates a new NormalDistParams instance with the given parameters.
func NewNormalDistParams(mean, stdDev, low, high float64) *NormalDistParams {
	return &NormalDistParams{
		Mean:   mean,
		StdDev: stdDev,
		Low:    low,
		High:   high,
	}
}

// Validate checks if the parameters are valid.
func (p *NormalDistParams) Validate() error {
	if p.Mean < p.Low || p.Mean > p.High {
		return fmt.Errorf("mean must be between low and high")
	}
	if p.StdDev <= 0 {
		return fmt.Errorf("std_dev must be positive")
	}
	if p.Low > p.High {
		return fmt.Errorf("low must be less than or equal to high")
	}
	return nil
}

func (p *NormalDistParams) Generate() float64 {
	// Генерируем два случайных числа из равномерного распределения
	u1 := rand.Float64()
	u2 := rand.Float64()

	// Применяем метод Бокса-Мюллера для генерации нормального распределения
	z0 := math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)

	// вычисляем нормальное распределение
	z := z0*p.StdDev + p.Mean

	// проверяем, что значение находится в пределах заданных границ
	if z < p.Low {
		z = p.Low
	} else if z > p.High {
		z = p.High
	}

	return z
}

func (p *NormalDistParams) GenerateVector(v []float64) error {
	for i := range v {
		v[i] = p.Generate()
	}
	return nil
}

func (p *NormalDistParams) RandN(n int) []float64 {
	r := make([]float64, n)
	p.GenerateVector(r)
	return r
}
