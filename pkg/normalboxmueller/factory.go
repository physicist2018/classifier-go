package normalboxmueller

func NewDistribution(isnormal bool, mean float64, stdDev float64, low float64, high float64) BoxDistrib {
	if isnormal {
		return NewNormalDistParams(mean, stdDev, low, high)
	} else {
		return NewUniDistParams(low, high)
	}
}
