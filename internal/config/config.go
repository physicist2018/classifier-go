package config

import "flag"

type Config struct {
	GfUrban, GfSoot, GfDust          float64
	DeltaUrban, DeltaSoot, DeltaDust float64
	VariationCoefficient             float64
	InputDir                         string
	NumPoints                        int
	DoSmooth                         bool
	SigmaH, SigmaT                   int // numer of temporal and spatial bins
	Size                             int // kernel size
}

func Parse() *Config {
	cfg := &Config{}

	// define flags
	flag.Float64Var(&cfg.GfUrban, "gf-urban", 0.55e-4, "fluorescence capacity of urban aerosol")
	flag.Float64Var(&cfg.GfSoot, "gf-soot", 4e-4, "fluorescence capacity of soot aerosol")
	flag.Float64Var(&cfg.GfDust, "gf-dust", 0.3e-4, "fluorescence capacity of dust aerosol")
	flag.Float64Var(&cfg.DeltaUrban, "delta-urban", 0.05, "aerosol depolarization for urban aerosol")
	flag.Float64Var(&cfg.DeltaSoot, "delta-soot", 0.06, "aerosol depolarization for soot aerosol")
	flag.Float64Var(&cfg.DeltaDust, "delta-dust", 0.26, "aerosol depolarization for dust aerosol")
	flag.Float64Var(&cfg.VariationCoefficient, "var-coef", 0.1, "variation coefficient for aerosol parameters")
	flag.StringVar(&cfg.InputDir, "input-dir", "./", "input directory")
	flag.IntVar(&cfg.NumPoints, "num-points", 100, "number of data points to simulate")
	flag.BoolVar(&cfg.DoSmooth, "smooth", false, "apply smoothing to the data")
	flag.IntVar(&cfg.SigmaH, "sigma-h", 5, "spatial smoothing size in bins")
	flag.IntVar(&cfg.SigmaT, "sigma-t", 3, "temporal smoothing size in bins")
	flag.IntVar(&cfg.Size, "size", 7, "kernel size in bins")
	flag.Parse()

	return cfg
}
