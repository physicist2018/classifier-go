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
	gfuMin := cfg.GfUrban * (1.0 - cfg.VarCoefGf)
	gfuMax := cfg.GfUrban * (1.0 + cfg.VarCoefGf)
	genGfUrban := normalboxmueller.NewDistribution(cfg.NormalRandomizer, cfg.GfUrban, (gfuMax-gfuMin)/4, gfuMin, gfuMax)
	gfUs := genGfUrban.RandN(cfg.NumPoints)

	// Soot
	gfsMin := cfg.GfSmoke * (1.0 - cfg.VarCoefGf)
	gfsMax := cfg.GfSmoke * (1.0 + cfg.VarCoefGf)
	genGfSoot := normalboxmueller.NewDistribution(cfg.NormalRandomizer, cfg.GfSmoke, (gfsMax-gfsMin)/4, gfsMin, gfsMax)
	gfSs := genGfSoot.RandN(cfg.NumPoints)

	// Dust
	gfdMin := cfg.GfDust * (1.0 - cfg.VarCoefGf)
	gfdMax := cfg.GfDust * (1.0 + cfg.VarCoefGf)
	genGfDust := normalboxmueller.NewDistribution(cfg.NormalRandomizer, cfg.GfDust, (gfdMax-gfdMin)/4, gfdMin, gfdMax)
	gfDs := genGfDust.RandN(cfg.NumPoints)

	// Delta distributions
	deltaSMin := cfg.DeltaSmoke * (1.0 - cfg.VarCoefDelta)
	deltaSMax := cfg.DeltaSmoke * (1.0 + cfg.VarCoefDelta)
	genDeltaSmoke := normalboxmueller.NewDistribution(cfg.NormalRandomizer, cfg.DeltaSmoke, (deltaSMax-deltaSMin)/4, deltaSMin, deltaSMax)
	deltaSs := genDeltaSmoke.RandN(cfg.NumPoints)

	deltaUMin := cfg.DeltaUrban * (1.0 - cfg.VarCoefDelta)
	deltaUMax := cfg.DeltaUrban * (1.0 + cfg.VarCoefDelta)
	genDeltaUrban := normalboxmueller.NewDistribution(cfg.NormalRandomizer, cfg.DeltaUrban, (deltaUMax-deltaUMin)/4, deltaUMin, deltaUMax)
	deltaUs := genDeltaUrban.RandN(cfg.NumPoints)

	deltaDMin := cfg.DeltaDust * (1.0 - cfg.VarCoefDelta)
	deltaDMax := cfg.DeltaDust * (1.0 + cfg.VarCoefDelta)
	genDeltaDust := normalboxmueller.NewDistribution(cfg.NormalRandomizer, cfg.DeltaDust, (deltaDMax-deltaDMin)/4, deltaDMin, deltaDMax)
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
