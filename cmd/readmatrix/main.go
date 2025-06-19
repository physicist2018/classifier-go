package main

import (
	"classifier-go/pkg/readmatrix"
	"gonum.org/v1/gonum/mat"
	"log"
)

func main() {
	m, err := readmatrix.ReadMatrix("1.txt")
	if err != nil {
		log.Fatal(err)
	}

	log.Println(m.Dims())
	log.Printf("a=%.2g", mat.Formatted(m, mat.Prefix(""), mat.Squeeze()))
}
