package normalboxmueller

import (
	"math/rand/v2"
)

type BoxDistrib interface {
	RandN(n int) []float64
}

type UniDistParams struct {
	//Mean   float64 `json:"mean"`
	//StdDev float64 `json:"std_dev"`
	Low  float64 `json:"low"`
	High float64 `json:"high"`
}

// NewNormalDistParams creates a new NormalDistParams instance with the given parameters.
func NewUniDistParams(low, high float64) *UniDistParams {
	return &UniDistParams{
		Low:  low,
		High: high,
	}
}

func (p *UniDistParams) Generate() float64 {
	// Генерируем два случайных числа из равномерного распределения
	span := p.High - p.Low
	v := rand.Float64()*span + p.Low

	return v
}

func (p *UniDistParams) GenerateVector(v []float64) error {
	for i := range v {
		v[i] = p.Generate()
	}
	return nil
}

func (p *UniDistParams) RandN(n int) []float64 {
	r := make([]float64, n)
	p.GenerateVector(r)
	return r
}
