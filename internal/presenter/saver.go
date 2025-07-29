package presenter

import (
	"encoding/csv"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func SaveDenseToCSV(m *mat.Dense, filename string) error {
	// Create the CSV file
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	rows, cols := m.Dims()

	// Write each row to the CSV file
	for i := 0; i < rows; i++ {
		record := make([]string, cols)
		for j := 0; j < cols; j++ {
			record[j] = strconv.FormatFloat(m.At(i, j), 'f', -1, 64)
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return nil
}
