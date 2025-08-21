package config

import (
	"encoding/json"
	"fmt"

	"github.com/spf13/pflag"
)

type Config struct {
	GfUrbanRange    []float64
	GfSmokeRange    []float64
	GfDustRange     []float64
	DeltaUrbanRange []float64
	DeltaSmokeRange []float64
	DeltaDustRange  []float64
	InputDir        string
	NumPoints       int
	DoSmooth        bool
	SigmaH          int
	SigmaT          int
	Size            int
	AvgPercent      float64
}

func Parse() *Config {
	// Создаем конфиг с дефолтными значениями
	cfg := &Config{
		GfUrbanRange:    []float64{0.1e-4, 1e-4},
		GfSmokeRange:    []float64{2e-4, 6e-4},
		GfDustRange:     []float64{0.1e-4, 0.5e-4},
		DeltaUrbanRange: []float64{0.01, 0.10},
		DeltaSmokeRange: []float64{0.02, 0.10},
		DeltaDustRange:  []float64{0.2, 0.35},
		InputDir:        "./",
		NumPoints:       100,
		DoSmooth:        false,
		SigmaH:          5,
		SigmaT:          3,
		Size:            7,
		AvgPercent:      0.1,
	}

	// Теперь парсим все флаги, которые могут переопределить значения
	parseFlags(cfg)

	return cfg
}

func parseFlags(cfg *Config) {
	pflag.Float64SliceVar(&cfg.GfUrbanRange, "gf-urban", cfg.GfUrbanRange, "емкость флуоресценции для городского аэрозоля")
	pflag.Float64SliceVar(&cfg.GfSmokeRange, "gf-smoke", cfg.GfSmokeRange, "емкость флуоресценции для смога")
	pflag.Float64SliceVar(&cfg.GfDustRange, "gf-dust", cfg.GfDustRange, "емкость флуоресценции для пылевого аэрозоля")
	pflag.Float64SliceVar(&cfg.DeltaUrbanRange, "delta-urban", cfg.DeltaUrbanRange, "аэрозольная деполяризация для городского аэрозоля")
	pflag.Float64SliceVar(&cfg.DeltaSmokeRange, "delta-smoke", cfg.DeltaSmokeRange, "аэрозольная деполяризация для смога")
	pflag.Float64SliceVar(&cfg.DeltaDustRange, "delta-dust", cfg.DeltaDustRange, "аэрозольная деполяризация для пылевого аэрозоля")
	pflag.StringVar(&cfg.InputDir, "input-dir", cfg.InputDir, "путь к каталогу где хранятся входные данные")
	pflag.IntVar(&cfg.NumPoints, "num-points", cfg.NumPoints, "число начальных значений реперных параметров")
	pflag.BoolVar(&cfg.DoSmooth, "smooth", cfg.DoSmooth, "применить сглаживание данных")
	pflag.IntVar(&cfg.SigmaH, "sigma-h", cfg.SigmaH, "полуширина окна сглаживания по высоте (отсчеты)")
	pflag.IntVar(&cfg.SigmaT, "sigma-t", cfg.SigmaT, "полуширина окна сглаживания по времени (отсчеты)")
	pflag.IntVar(&cfg.Size, "size", cfg.Size, "размер окна сглаживания")
	pflag.Float64Var(&cfg.AvgPercent, "avg-percent", cfg.AvgPercent, "персентиль для усреднения данных")

	pflag.Parse()
}

func (c *Config) ToString() string {
	ret, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	return string(ret)
}
