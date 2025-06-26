package heatmapplotter

import (
	"fmt"
	"log"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgpdf"
)

func MakeHeatmapPlot(data *mat.Dense, title, filename string) {
	p := plot.New()
	p.Title.Text = title
	p.X.Label.Text = "Time, (profile no)"
	p.Y.Label.Text = "Altitude, (m)"

	pal := palette.Rainbow(10, palette.Blue, palette.Red, 1, 1, 1)
	heatmap := plotter.NewHeatMap(
		matrixToGrid(data), // Замените на вашу функцию преобразования
		pal,
	)
	heatmap.Max = 1
	heatmap.Min = 0

	p.Add(heatmap)

	// Create a legend.
	l := plot.NewLegend()
	thumbs := plotter.PaletteThumbnailers(pal)
	nthumbs := len(thumbs)
	for i := nthumbs - 1; i >= 0; i-- {
		t := thumbs[i]
		val := (heatmap.Max - heatmap.Min) / float64(nthumbs) * float64(i)
		if i != 0 && i != len(thumbs)-1 {
			l.Add(fmt.Sprintf("%.1f", val), t)
			continue
		}
		//var val float64
		switch i {
		case 0:
			val = heatmap.Min
		case len(thumbs) - 1:
			val = heatmap.Max
		}
		l.Add(fmt.Sprintf("%.1f", val), t)
	}

	p.X.Padding = 0
	p.Y.Padding = 0

	// Create a PDF
	img := vgpdf.New(vg.Points(400), vg.Points(200))
	//img := vgimg.New(250, 175)

	dc := draw.New(img)

	l.Top = true
	// Calculate the width of the legend.
	r := l.Rectangle(dc)
	legendWidth := r.Max.X - r.Min.X
	l.YOffs = -p.Title.TextStyle.FontExtents().Height // Adjust the legend down a little.

	l.Draw(dc)
	dc = draw.Crop(dc, 0, -legendWidth-vg.Millimeter, 0, 0) // Make space for the legend.
	p.Draw(dc)
	w, err := os.Create(filename)
	if err != nil {
		log.Panic(err)
	}
	defer w.Close()

	//png := vgimg.PngCanvas{Canvas: img}
	if _, err = img.WriteTo(w); err != nil {
		log.Panic(err)
	}
}

func matrixToGrid(matrix *mat.Dense) plotter.GridXYZ {
	return grid{
		Matrix: matrix,
		Rows:   matrix.RawMatrix().Rows,
		Cols:   matrix.RawMatrix().Cols,
	}
}

type grid struct {
	Matrix     *mat.Dense
	Rows, Cols int
}

func (g grid) Dims() (c, r int)   { return g.Cols - 1, g.Rows }
func (g grid) Z(c, r int) float64 { return g.Matrix.At(r, c+1) }
func (g grid) X(c int) float64    { return float64(c) }
func (g grid) Y(r int) float64    { return g.Matrix.At(r, 0) }
