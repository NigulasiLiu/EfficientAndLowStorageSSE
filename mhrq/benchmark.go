package mhrq

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

func sortKeywords(invertedIndex map[string][]int) []string {
	keywords := make([]string, 0, len(invertedIndex))
	for keyword := range invertedIndex {
		keywords = append(keywords, keyword)
	}
	sort.Slice(keywords, func(i, j int) bool {
		ki, _ := strconv.ParseInt(keywords[i], 10, 64)
		kj, _ := strconv.ParseInt(keywords[j], 10, 64)
		return ki < kj
	})
	return keywords
}

func loadInvertedIndex(filePath string) (map[string][]int, error) {
	invertedIndex := make(map[string][]int)
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("无法打开文件: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) < 2 {
			return nil, fmt.Errorf("无效的行格式: %s", line)
		}
		keyword := parts[0]
		rowIDs := make([]int, 0, len(parts)-1)
		for _, part := range parts[1:] {
			rowID, err := strconv.Atoi(part)
			if err != nil {
				continue
			}
			rowIDs = append(rowIDs, rowID)
		}
		invertedIndex[keyword] = rowIDs
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("读取文件出错: %v", err)
	}
	return invertedIndex, nil
}

func generateQueryRangeWithWidth(keywords []string, width int) ([2]string, int) {
	n := len(keywords)
	if n < 2 || width <= 0 {
		return [2]string{"0", "0"}, 0
	}
	for attempts := 0; attempts < 10*n; attempts++ {
		i := rand.Intn(n)
		left, err := strconv.Atoi(keywords[i])
		if err != nil {
			continue
		}
		right := left + width
		maxKeyword, err := strconv.Atoi(keywords[n-1])
		if err != nil {
			break
		}
		if right <= maxKeyword {
			return [2]string{strconv.Itoa(left), strconv.Itoa(right)}, width
		}
	}
	left, _ := strconv.Atoi(keywords[0])
	return [2]string{strconv.Itoa(left), strconv.Itoa(left + width)}, width
}

func writeCSVHeader(w *csv.Writer) error {
	return w.Write([]string{"phase", "dataset_keywords", "range_width", "iteration", "build_time_ns", "query_time_ns", "update_time_ns", "storage_bytes", "tokens", "results"})
}

func estimateStorageBytes(s *Scheme) int {
	bytes := 0
	if s.setup != nil {
		bytes += len(s.setup.KSE) + len(s.setup.KPRF)
	}
	for _, list := range s.edb {
		for _, ct := range list {
			bytes += len(ct.ID) + len(ct.Keyword) + len(ct.ST) + len(ct.PrevST) + len(ct.Address) + len(ct.Delta) + len(ct.Token)
			if ct.EncryptedP != nil {
				bytes += ct.EncryptedP.N * ct.EncryptedP.N * 8
			}
		}
	}
	return bytes
}
