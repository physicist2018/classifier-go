package randomnormal

import (
	"gonum.org/v1/gonum/stat/distuv"
	"math/rand"
	"time"
)

// NormalRandGenerator генерирует случайные числа с нормальным распределением
// в заданном интервале [min, max]
type NormalRandGenerator struct {
	dist distuv.Normal
	min  float64
	max  float64
}

// NewNormalRandGenerator создает новый генератор
func NewNormalRandGenerator(mean, stddev, min, max float64) *NormalRandGenerator {
	if min >= max {
		panic("min must be less than max")
	}
	src := rand.NewSource(time.Now().UnixNano())
	rnd := rand.New(src)
	return &NormalRandGenerator{
		dist: distuv.Normal{
			Mu:    mean,
			Sigma: stddev,
			Src:   rnd,
		},
		min: min,
		max: max,
	}
}

// Rand генерирует одно случайное число в заданном интервале
func (g *NormalRandGenerator) Rand() float64 {
	for {
		val := g.dist.Rand()
		if val >= g.min && val <= g.max {
			return val
		}
	}
}

// RandN генерирует n случайных чисел в заданном интервале
func (g *NormalRandGenerator) RandN(n int) []float64 {
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = g.Rand()
	}
	return result
}

func (g *NormalRandGenerator) Min() float64 {
	return g.min
}

func (g *NormalRandGenerator) Max() float64 {
	return g.max
}
