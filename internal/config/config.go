package config

import (
	"github.com/spf13/pflag"
)

type Config struct {
	GfUrban      float64
	GfSmoke      float64
	GfDust       float64
	DeltaUrban   float64
	DeltaSmoke   float64
	DeltaDust    float64
	VarCoefDelta float64
	VarCoefGf    float64
	InputDir     string
	NumPoints    int
	DoSmooth     bool
	SigmaH       int
	SigmaT       int
	Size         int
	AvgPercent   float64
}

func Parse() *Config {
	// Создаем конфиг с дефолтными значениями
	cfg := &Config{
		GfUrban:      0.55e-4,
		GfSmoke:      4e-4,
		GfDust:       0.3e-4,
		DeltaUrban:   0.05,
		DeltaSmoke:   0.06,
		DeltaDust:    0.26,
		VarCoefDelta: 0.1,
		VarCoefGf:    0.1,
		InputDir:     "./",
		NumPoints:    100,
		DoSmooth:     false,
		SigmaH:       5,
		SigmaT:       3,
		Size:         7,
		AvgPercent:   0.1,
	}

	// Теперь парсим все флаги, которые могут переопределить значения
	parseFlags(cfg)

	return cfg
}

func parseFlags(cfg *Config) {
	pflag.Float64Var(&cfg.GfUrban, "gf-urban", cfg.GfUrban, "емкость флуоресценции для городского аэрозоля")
	pflag.Float64Var(&cfg.GfSmoke, "gf-smoke", cfg.GfSmoke, "ескость флкоресценции для смога")
	pflag.Float64Var(&cfg.GfDust, "gf-dust", cfg.GfDust, "ескость флуоресценции для пылевого аэрозоля")
	pflag.Float64Var(&cfg.DeltaUrban, "delta-urban", cfg.DeltaUrban, "аэрозольная деполяризация для городского аэрозоля")
	pflag.Float64Var(&cfg.DeltaSmoke, "delta-smoke", cfg.DeltaSmoke, "аэрозольная деполяризация для смога")
	pflag.Float64Var(&cfg.DeltaDust, "delta-dust", cfg.DeltaDust, "аэрозольная деполяризация для пылевого аэрозоля")
	pflag.Float64Var(&cfg.VarCoefDelta, "var-coef-delta", cfg.VarCoefDelta, "параметр вариативности для деполяризации")
	pflag.Float64Var(&cfg.VarCoefGf, "var-coef-gf", cfg.VarCoefGf, "параметр вариативности для емкости флуоресценции")
	pflag.StringVar(&cfg.InputDir, "input-dir", cfg.InputDir, "путь к каталогу где хранятся входные данные")
	pflag.IntVar(&cfg.NumPoints, "num-points", cfg.NumPoints, "число начальных значений реперных параметров")
	pflag.BoolVar(&cfg.DoSmooth, "smooth", cfg.DoSmooth, "применить сглаживание данных")
	pflag.IntVar(&cfg.SigmaH, "sigma-h", cfg.SigmaH, "полуширина окна сглаживания по высоте (отсчеты)")
	pflag.IntVar(&cfg.SigmaT, "sigma-t", cfg.SigmaT, "полуширина окна сглаживания по времени (отсчеты)")
	pflag.IntVar(&cfg.Size, "size", cfg.Size, "размер окна сглаживания")
	pflag.Float64Var(&cfg.AvgPercent, "avg-percent", cfg.AvgPercent, "персентиль для усреднения данных")

	pflag.Parse()
}
