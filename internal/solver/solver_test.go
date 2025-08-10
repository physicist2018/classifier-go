package prepare

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

const Tolerance = 1e-3

func Equals(a, b float64) bool {
	delta := math.Abs(a - b)
	if delta < Tolerance {
		return true
	}
	return false
}

// TestAveragePRows tests the AveragePRows function.
func TestAveragePRows(t *testing.T) {
	A := mat.NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	result := AveragePRows(A, 2)
	flag := Equals(result.AtVec(0), 2) && Equals(result.AtVec(1), 3)
	if !flag {
		t.Errorf("Expected flag to be true, got %v", flag)
	}
}

// TestSolveWithBounds tests the SolveWithBounds function.
func TestSolveWithBounds(t *testing.T) {

	A := mat.NewDense(3, 3, []float64{1, 2, 3, 4, 6, 6, 7, 88, 9})
	b := mat.NewVecDense(3, []float64{1, 2, 3})

	x, F := SolveWithBounds(A, b, []float64{-1, -1, -1}, []float64{1, 1, 1})
	if !Equals(F, 0) {
		t.Errorf("Expected F to be 0, got %v", F)
	}
	if !Equals(x.AtVec(0), 0) || !Equals(x.AtVec(1), 0) || !Equals(x.AtVec(2), 0.333333) {
		t.Errorf("Expected x to be [0.0, 0.0, 0.33333333], got %v", x)
	}

}
