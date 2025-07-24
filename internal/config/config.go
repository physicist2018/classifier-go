package config

import (
	"flag"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	GfUrban              float64 `yaml:"gf_urban"`
	GfSoot               float64 `yaml:"gf_soot"`
	GfDust               float64 `yaml:"gf_dust"`
	DeltaUrban           float64 `yaml:"delta_urban"`
	DeltaSoot            float64 `yaml:"delta_soot"`
	DeltaDust            float64 `yaml:"delta_dust"`
	VariationCoefficient float64 `yaml:"variation_coefficient"`
	InputDir             string  `yaml:"input_dir"`
	NumPoints            int     `yaml:"num_points"`
	DoSmooth             bool    `yaml:"do_smooth"`
	SigmaH               int     `yaml:"sigma_h"`
	SigmaT               int     `yaml:"sigma_t"`
	Size                 int     `yaml:"size"`
	AvgPercent           float64 `yaml:"avg_percent"`
}

func Parse() *Config {
	// Сначала создаем временный парсер только для флага config
	tempFlags := flag.NewFlagSet("temp", flag.ContinueOnError)
	configFile := tempFlags.String("config", "config.yml", "path to config file")

	// Парсим только флаг config, игнорируя другие
	tempFlags.Parse(os.Args[1:])

	// Создаем конфиг с дефолтными значениями
	cfg := &Config{
		GfUrban:              0.55e-4,
		GfSoot:               4e-4,
		GfDust:               0.3e-4,
		DeltaUrban:           0.05,
		DeltaSoot:            0.06,
		DeltaDust:            0.26,
		VariationCoefficient: 0.1,
		InputDir:             "./",
		NumPoints:            100,
		DoSmooth:             false,
		SigmaH:               5,
		SigmaT:               3,
		Size:                 7,
		AvgPercent:           0.1,
	}

	// Загружаем конфиг из YAML, если файл существует
	if _, err := os.Stat(*configFile); err == nil {
		if err := loadFromYAML(*configFile, cfg); err != nil {
			fmt.Printf("Error loading config from YAML: %v\n", err)
		}
	}

	// Теперь парсим все флаги, которые могут переопределить значения
	parseFlags(cfg)

	return cfg
}

func loadFromYAML(filename string, cfg *Config) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("failed to read config file: %w", err)
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return fmt.Errorf("failed to unmarshal YAML: %w", err)
	}

	return nil
}

func parseFlags(cfg *Config) {
	flag.Float64Var(&cfg.GfUrban, "gf-urban", cfg.GfUrban, "fluorescence capacity of urban aerosol")
	flag.Float64Var(&cfg.GfSoot, "gf-soot", cfg.GfSoot, "fluorescence capacity of soot aerosol")
	flag.Float64Var(&cfg.GfDust, "gf-dust", cfg.GfDust, "fluorescence capacity of dust aerosol")
	flag.Float64Var(&cfg.DeltaUrban, "delta-urban", cfg.DeltaUrban, "aerosol depolarization for urban aerosol")
	flag.Float64Var(&cfg.DeltaSoot, "delta-soot", cfg.DeltaSoot, "aerosol depolarization for soot aerosol")
	flag.Float64Var(&cfg.DeltaDust, "delta-dust", cfg.DeltaDust, "aerosol depolarization for dust aerosol")
	flag.Float64Var(&cfg.VariationCoefficient, "var-coef", cfg.VariationCoefficient, "variation coefficient for aerosol parameters")
	flag.StringVar(&cfg.InputDir, "input-dir", cfg.InputDir, "input directory")
	flag.IntVar(&cfg.NumPoints, "num-points", cfg.NumPoints, "number of data points to simulate")
	flag.BoolVar(&cfg.DoSmooth, "smooth", cfg.DoSmooth, "apply smoothing to the data")
	flag.IntVar(&cfg.SigmaH, "sigma-h", cfg.SigmaH, "spatial smoothing size in bins")
	flag.IntVar(&cfg.SigmaT, "sigma-t", cfg.SigmaT, "temporal smoothing size in bins")
	flag.IntVar(&cfg.Size, "size", cfg.Size, "kernel size in bins")
	flag.Float64Var(&cfg.AvgPercent, "avg-percent", cfg.AvgPercent, "percentage of data to average")

	flag.Parse()
}
