package mhrq

import (
	"bufio"
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
		filepath.Join("dataset", "Gowalla_invertedIndex_new_25000.txt"),
	}
	indexNum := []int{5000, 10000, 15000, 20000, 25000}
	ranges := []int{600, 1200, 1800, 2400, 3000, 3600, 4200, 4800}
	LValues := []int{6424}
	// Keep query attempts at hundred/thousand level for practical runtime.
	k := 1000
	resultCounts := 100
	resultsDir := filepath.Join("results", "mhrq")
	if err := os.MkdirAll(resultsDir, 0o755); err != nil {
		return err
	}

	progressPath := filepath.Join(resultsDir, "mhrq_comparison_progress.log")
	progressFile, err := os.Create(progressPath)
	if err != nil {
		return err
	}
	defer progressFile.Close()
	progressW := bufio.NewWriter(progressFile)
	defer progressW.Flush()
	_, _ = progressW.WriteString(fmt.Sprintf("[%s] [MHRQ-Comparison] start\n", nowStamp()))

	totalDatasets := len(files)
	for fileIndex, file := range files {
		datasetProgress := int(float64(fileIndex) / float64(totalDatasets) * 100)
		fmt.Printf("[MHRQ Comparison] Progress %d%% - loading dataset (%d/%d): %s\n", datasetProgress, fileIndex+1, totalDatasets, file)
		resolved := resolveDatasetPath(file)
		_, _ = progressW.WriteString(fmt.Sprintf("[%s] [Dataset] loading %s (resolved=%s)\n", nowStamp(), file, resolved))
		if _, err := os.Stat(resolved); err != nil {
			return fmt.Errorf("dataset not found: %s", resolved)
		}
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
			_, _ = progressW.WriteString(fmt.Sprintf("[%s] [Build] m=%d start\n", nowStamp(), indexNum[fileIndex]))
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
			_, _ = progressW.WriteString(fmt.Sprintf("[%s] [Build] m=%d done duration_ns=%d\n", nowStamp(), indexNum[fileIndex], buildDuration))

			outPath := filepath.Join(resultsDir, fmt.Sprintf("mhrq_comparison_config1_m_%d.txt", indexNum[fileIndex]))
			f, err := os.Create(outPath)
			if err != nil {
				return err
			}
			writer := bufio.NewWriter(f)
			if err := writeComparisonTXTHeader(writer); err != nil {
				_ = f.Close()
				return err
			}
			for rIdx, r := range ranges {
				overallUnits := totalDatasets * len(ranges)
				completedUnits := fileIndex*len(ranges) + rIdx
				rangeProgress := int(float64(completedUnits) / float64(overallUnits) * 100)
				fmt.Printf("[MHRQ Comparison] Progress %d%% - search m=%d range=%d (%d/%d)\n", rangeProgress, indexNum[fileIndex], r, rIdx+1, len(ranges))
				_, _ = progressW.WriteString(fmt.Sprintf("[%s] [Search] m=%d range=%d (%d/%d) start\n", nowStamp(), indexNum[fileIndex], r, rIdx+1, len(ranges)))
				validCount := 0
				for i := 0; i < k; i++ {
					if (i+1)%100 == 0 {
						fmt.Printf("[MHRQ Comparison] Progress %d%% - search m=%d range=%d loop=%d/%d valid=%d\n", rangeProgress, indexNum[fileIndex], r, i+1, k, validCount)
					}
					queryRange, rangeWidth := generateQueryRangeWithWidth(sortedKeywords, r)
					startQuery := time.Now()
					res, err := s.Search(queryRange[0], atoiSafe(queryRange[0]), atoiSafe(queryRange[1]))
					queryDuration := time.Since(startQuery).Nanoseconds()
					if err != nil {
						_ = f.Close()
						return err
					}
					if len(res) == 0 {
						_, _ = writer.WriteString(fmt.Sprintf("search\t%d\t%d\t%d\t%d\t%d\t0\t%d\t%d\t0\n", len(sortedKeywords), rangeWidth, i+1, buildDuration, queryDuration, estimateStorageBytes(s), searchTokenCountByPseudoCode()))
						continue
					}
					validCount++
					_, _ = writer.WriteString(fmt.Sprintf("search\t%d\t%d\t%d\t%d\t%d\t0\t%d\t%d\t%d\n", len(sortedKeywords), rangeWidth, i+1, buildDuration, queryDuration, estimateStorageBytes(s), searchTokenCountByPseudoCode(), len(res)))
					if validCount >= resultCounts {
						break
					}
				}
				_, _ = progressW.WriteString(fmt.Sprintf("[%s] [Search] m=%d range=%d done valid=%d\n", nowStamp(), indexNum[fileIndex], r, validCount))
				_ = writer.Flush()
				_ = progressW.Flush()
			}
			_ = writer.Flush()
			_ = f.Close()
		}
	}
	fmt.Println("[MHRQ Comparison] Progress 100% - finished")
	_, _ = progressW.WriteString(fmt.Sprintf("[%s] [MHRQ-Comparison] finished\n", nowStamp()))
	return nil
}

func atoiSafe(s string) int {
	v, _ := strconv.Atoi(s)
	return v
}
