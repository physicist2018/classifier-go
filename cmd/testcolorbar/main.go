package main

import (
	"fmt"
	"log"
	"math"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

type grid struct {
	data        [][]float64
	xmin, xstep float64
	ymin, ystep float64
}

func (g grid) Dims() (c, r int)   { return len(g.data[0]), len(g.data) }
func (g grid) Z(c, r int) float64 { return g.data[r][c] }
func (g grid) X(c int) float64    { return g.xmin + float64(c)*g.xstep }
func (g grid) Y(r int) float64    { return g.ymin + float64(r)*g.ystep }

func main() {
	// Генерация данных
	nx, ny := 100, 100
	data := make([][]float64, ny)
	for y := 0; y < ny; y++ {
		data[y] = make([]float64, nx)
		for x := 0; x < nx; x++ {
			data[y][x] = math.Sin(float64(x)/10) * math.Cos(float64(y)/10)
		}
	}

	grid := grid{
		data:  data,
		xmin:  0,
		xstep: 1,
		ymin:  0,
		ystep: 1,
	}

	pal := palette.Heat(12, 1)
	h := plotter.NewHeatMap(grid, pal)

	p := plot.New()
	p.Title.Text = "Heat map"

	p.Add(h)

	// Create a legend.
	l := plot.NewLegend()
	thumbs := plotter.PaletteThumbnailers(pal)
	for i := len(thumbs) - 1; i >= 0; i-- {
		t := thumbs[i]
		if i != 0 && i != len(thumbs)-1 {
			l.Add("", t)
			continue
		}
		var val float64
		switch i {
		case 0:
			val = h.Min
		case len(thumbs) - 1:
			val = h.Max
		}
		l.Add(fmt.Sprintf("%.2g", val), t)
	}

	p.X.Padding = 0
	p.Y.Padding = 0
	p.X.Max = 1.5
	p.Y.Max = 1.5

	img := vgimg.New(250, 175)
	dc := draw.New(img)

	l.Top = true
	// Calculate the width of the legend.
	r := l.Rectangle(dc)
	legendWidth := r.Max.X - r.Min.X
	l.YOffs = -p.Title.TextStyle.FontExtents().Height // Adjust the legend down a little.

	l.Draw(dc)
	dc = draw.Crop(dc, 0, -legendWidth-vg.Millimeter, 0, 0) // Make space for the legend.
	p.Draw(dc)
	w, err := os.Create("heatMap.png")
	if err != nil {
		log.Panic(err)
	}
	png := vgimg.PngCanvas{Canvas: img}
	if _, err = png.WriteTo(w); err != nil {
		log.Panic(err)
	}

}
