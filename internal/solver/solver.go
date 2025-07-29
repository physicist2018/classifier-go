package prepare

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

func SolveWithBounds(A *mat.Dense, b *mat.VecDense, lower, upper []float64) (*mat.VecDense, float64) {
	_, n := A.Dims()

	problem := optimize.Problem{
		Func: func(x []float64) float64 {
			var Ax mat.VecDense
			Ax.MulVec(A, mat.NewVecDense(n, x))
			var diff mat.VecDense
			diff.SubVec(&Ax, b)
			penalty := 0
			for i := range len(x) {
				if x[i] < 0 || x[i] > 1.0 {
					penalty += 1000
				}
			}
			if floats.Sum(x) > 1.1 {
				penalty += 1000
			}

			return mat.Norm(&diff, 2) + float64(penalty)
		},
	}

	settings := &optimize.Settings{
		MajorIterations: 1000,
		FuncEvaluations: 1000,
	}

	result, err := optimize.Minimize(problem, make([]float64, n), settings, &optimize.NelderMead{})
	if err != nil {
		panic(err)
	}

	F := problem.Func(result.X)

	return mat.NewVecDense(n, result.X), F
}

func AveragePRows(A *mat.Dense, n int) *mat.VecDense {
	r, c := A.Dims()
	if n > r {
		n = r
	}

	sum := mat.NewVecDense(c, nil)
	for i := range n {
		row := A.RowView(i)
		sum.AddVec(sum, row)
	}
	sum.ScaleVec(1/float64(n), sum)
	return sum
}
