package main

import (
	"classifier-go/pkg/convolve"
	"classifier-go/pkg/heatmapplotter"
	"classifier-go/pkg/readmatrix"
	"log"

	"gonum.org/v1/gonum/mat"
)

func main() {
	m, err := readmatrix.ReadMatrix("1.txt")
	rows, cols := m.Dims()
	data := m.Slice(0, rows, 1, cols).(*mat.Dense)

	if err != nil {
		log.Fatal(err)
	}

	data.Scale(0.01, data)

	gs := convolve.NewGaussianKernel(5, 5, 7)
	output := gs.Convolve(data)
	data.Copy(output)

	heatmapplotter.MakeHeatmapPlot(m, "InitaialData", "1.pdf")

}
