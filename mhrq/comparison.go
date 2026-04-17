package mhrq

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

// RunComparison mirrors the style of the repository's existing comparison driver.
func RunComparison() error {
	files := []string{
		filepath.Join("dataset", "Gowalla_invertedIndex_new_5000.txt"),
		filepath.Join("dataset", "Gowalla_invertedIndex_new_10000.txt"),
		filepath.Join("dataset", "Gowalla_invertedIndex_new_15000.txt"),
		filepath.Join("dataset", "Gowalla_invertedIndex_new_20000.txt"),
	}
	indexNum := []int{5000, 10000, 15000, 20000}
	ranges := []int{600, 1200, 1800, 2400, 3000, 3600, 4200, 4800}
	LValues := []int{6424}
	k := 999999
	resultCounts := 200
	resultsDir := filepath.Join("results", "mhrq")
	if err := os.MkdirAll(resultsDir, 0o755); err != nil {
		return err
	}

	for fileIndex, file := range files {
		invertedIndex, err := loadInvertedIndex(file)
		if err != nil {
			return fmt.Errorf("load %s: %w", file, err)
		}
		sortedKeywords := sortKeywords(invertedIndex)
		if len(sortedKeywords) == 0 {
			continue
		}
		for _, L := range LValues {
			_ = L
			s, err := Setup(128, 32)
			if err != nil {
				return err
			}
			start := time.Now()
			for _, keyword := range sortedKeywords {
				for _, docID := range invertedIndex[keyword] {
					_, err := s.Update(strconv.Itoa(docID), "add", keyword, docID)
					if err != nil {
						return err
					}
				}
			}
			buildDuration := time.Since(start).Nanoseconds()

			outPath := filepath.Join(resultsDir, fmt.Sprintf("comparison_result_m_%d_mhrq.csv", indexNum[fileIndex]))
			f, err := os.Create(outPath)
			if err != nil {
				return err
			}
			writer := csv.NewWriter(f)
			if err := writeCSVHeader(writer); err != nil {
				_ = f.Close()
				return err
			}
			for _, r := range ranges {
				validCount := 0
				for i := 0; i < k; i++ {
					queryRange, rangeWidth := generateQueryRangeWithWidth(sortedKeywords, r)
					startQuery := time.Now()
					res, err := s.Search(queryRange[0], atoiSafe(queryRange[0]), atoiSafe(queryRange[1]))
					queryDuration := time.Since(startQuery).Nanoseconds()
					if err != nil {
						_ = f.Close()
						return err
					}
					if len(res) == 0 {
						_ = writer.Write([]string{"search", strconv.Itoa(len(sortedKeywords)), strconv.Itoa(rangeWidth), strconv.Itoa(i + 1), strconv.FormatInt(buildDuration, 10), strconv.FormatInt(queryDuration, 10), "0", strconv.Itoa(estimateStorageBytes(s)), "0", "0"})
						continue
					}
					validCount++
					_ = writer.Write([]string{"search", strconv.Itoa(len(sortedKeywords)), strconv.Itoa(rangeWidth), strconv.Itoa(i + 1), strconv.FormatInt(buildDuration, 10), strconv.FormatInt(queryDuration, 10), "0", strconv.Itoa(estimateStorageBytes(s)), strconv.Itoa(2), strconv.Itoa(len(res))})
					if validCount >= resultCounts {
						break
					}
				}
			}
			writer.Flush()
			_ = f.Close()
		}
	}
	return nil
}

func atoiSafe(s string) int {
	v, _ := strconv.Atoi(s)
	return v
}
