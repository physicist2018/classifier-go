package readmatrix

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

func ReadMatrix(filename string) (*mat.Dense, error) {
	// Открываем файл
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	var rows [][]float64
	scanner := bufio.NewScanner(file)
	hasHeader := false

	for scanner.Scan() {
		line := scanner.Text()

		// Пропускаем пустые строки и строки с комментариями
		if len(line) == 0 || strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}

		// Разбиваем строку на поля (разделители: табуляция или пробелы)
		fields := strings.Fields(line)

		// Проверяем, есть ли заголовок (если первая строка содержит нечисловые значения)
		if !hasHeader {
			allNumeric := true
			for _, field := range fields {
				if _, err := strconv.ParseFloat(field, 64); err != nil {
					allNumeric = false
					break
				}
			}

			if !allNumeric {
				hasHeader = true
				continue // Пропускаем строку с заголовком
			}
		}

		// Парсим числовые данные
		row := make([]float64, len(fields))
		for i, field := range fields {
			val, err := strconv.ParseFloat(field, 64)
			if err != nil {
				return nil, fmt.Errorf("failed to parse float at line %d, column %d: %v",
					len(rows)+1, i+1, err)
			}
			row[i] = val
		}

		rows = append(rows, row)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %v", err)
	}

	if len(rows) == 0 {
		return mat.NewDense(0, 0, nil), nil
	}

	// Создаём матрицу
	cols := len(rows[0])
	flatData := make([]float64, 0, len(rows)*cols)

	for _, row := range rows {
		if len(row) != cols {
			return nil, fmt.Errorf("inconsistent number of columns: expected %d, got %d",
				cols, len(row))
		}
		flatData = append(flatData, row...)
	}

	return mat.NewDense(len(rows), cols, flatData), nil
}
