package prepare

import (
	"classifier-go/internal/config"
	"classifier-go/pkg/normalboxmueller"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// PrepareMatrixA - подготавливает ядра СЛАУ с учетм вариативности параметров в настройках
func PrepareMatricesA(cfg *config.Config) []*mat.Dense {
	// Urban
	genGfUrban := normalboxmueller.NewUniDistParams(cfg.GfUrbanRange[0], cfg.GfUrbanRange[1])
	gfUs := genGfUrban.RandN(cfg.NumPoints)

	// Soot
	genGfSoot := normalboxmueller.NewUniDistParams(cfg.GfSmokeRange[0], cfg.GfSmokeRange[1])
	gfSs := genGfSoot.RandN(cfg.NumPoints)

	// Dust
	genGfDust := normalboxmueller.NewUniDistParams(cfg.GfDustRange[0], cfg.GfDustRange[1])
	gfDs := genGfDust.RandN(cfg.NumPoints)

	// Delta distributions

	deltaSMin := cfg.DeltaSmokeRange[0] / (1.0 + cfg.DeltaSmokeRange[0])
	deltaSMax := cfg.DeltaSmokeRange[1] / (1.0 + cfg.DeltaSmokeRange[1])
	genDeltaSmoke := normalboxmueller.NewUniDistParams(deltaSMin, deltaSMax)
	deltaSs := genDeltaSmoke.RandN(cfg.NumPoints)

	deltaUMin := cfg.DeltaUrbanRange[0] / (1.0 + cfg.DeltaUrbanRange[0])
	deltaUMax := cfg.DeltaUrbanRange[1] / (1.0 + cfg.DeltaUrbanRange[1])
	genDeltaUrban := normalboxmueller.NewUniDistParams(deltaUMin, deltaUMax)
	deltaUs := genDeltaUrban.RandN(cfg.NumPoints)

	deltaDMin := cfg.DeltaDustRange[0] / (1.0 + cfg.DeltaDustRange[0])
	deltaDMax := cfg.DeltaDustRange[1] / (1.0 + cfg.DeltaDustRange[1])
	genDeltaDust := normalboxmueller.NewUniDistParams(deltaDMin, deltaDMax)
	deltaDs := genDeltaDust.RandN(cfg.NumPoints)

	res := make([]*mat.Dense, cfg.NumPoints)
	for i := range cfg.NumPoints {
		res[i] = mat.NewDense(3, 3, []float64{deltaDs[i], deltaUs[i], deltaSs[i], gfDs[i], gfUs[i], gfSs[i], 1, 1, 1})
	}
	return res
}

func FormBVector(delta, gf float64) *mat.VecDense {
	return mat.NewVecDense(3, []float64{delta, gf, 1})
}

// RowSlice attaches sort.Interface to a slice of matrix rows.
type RowSlice struct {
	Matrix *mat.Dense
	Rows   []int // Row indices to sort
	Col    int   // Column to use for comparison
}

func (r RowSlice) Len() int      { return len(r.Rows) }
func (r RowSlice) Swap(i, j int) { r.Rows[i], r.Rows[j] = r.Rows[j], r.Rows[i] }
func (r RowSlice) Less(i, j int) bool {
	return r.Matrix.At(r.Rows[i], r.Col) < r.Matrix.At(r.Rows[j], r.Col)
}

// SortRows sorts the rows of a matrix in-place based on values in column `col`.
func SortRows(m *mat.Dense, col int) {
	rows, _ := m.Dims()
	rowIndices := make([]int, rows)
	for i := range rowIndices {
		rowIndices[i] = i
	}

	sort.Sort(RowSlice{m, rowIndices, col})

	// Reconstruct the matrix with sorted rows
	sorted := mat.NewDense(rows, m.RawMatrix().Cols, nil)
	for i, row := range rowIndices {
		sorted.SetRow(i, m.RawRowView(row))
	}

	*m = *sorted
}
