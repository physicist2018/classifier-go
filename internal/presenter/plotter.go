package presenter

import (
	"classifier-go/pkg/heatmapplotter"

	"gonum.org/v1/gonum/mat"
)

func GenerateHeatmap(outputPath string, title string, matrix *mat.Dense) {
	heatmapplotter.MakeHeatmapPlot(matrix, title, outputPath)
}
