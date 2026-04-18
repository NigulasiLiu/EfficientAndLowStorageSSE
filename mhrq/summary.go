package mhrq

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// GenerateMHRQSummaryTables writes table-ready outputs into results/mhrq by
// summarizing any existing experiment txt files.
func GenerateMHRQSummaryTables() error {
	return writeMHRQSummaryTables(filepath.Join("results", "mhrq"))
}

type mhrqSummaryRow struct {
	phase       string
	keywords    int
	rangeWidth  int
	buildNS     int64
	queryNS     int64
	updateNS    int64
	storageByte int
	tokens      int
	results     int
	durationNS  int64
}

func writeMHRQSummaryTables(resultsDir string) error {
	if err := os.MkdirAll(resultsDir, 0o755); err != nil {
		return err
	}

	rows, err := loadMHRQSummaryRows(resultsDir)
	if err != nil {
		return err
	}

	files := map[string]string{
		"update": filepath.Join(resultsDir, "mhrq_update_table_config1.txt"),
		"search": filepath.Join(resultsDir, "mhrq_search_tokens_table_config1.txt"),
		"build":  filepath.Join(resultsDir, "mhrq_build_table_config1.txt"),
	}

	if err := writeMHRQUpdateTable(files["update"], rows); err != nil {
		return err
	}
	if err := writeMHRQSearchTable(files["search"], rows); err != nil {
		return err
	}
	if err := writeMHRQBuildTable(files["build"], rows); err != nil {
		return err
	}
	return nil
}

func loadMHRQSummaryRows(resultsDir string) (map[int]map[int][]mhrqSummaryRow, error) {
	out := make(map[int]map[int][]mhrqSummaryRow)
	err := filepath.WalkDir(resultsDir, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		if filepath.Ext(path) != ".txt" {
			return nil
		}
		base := filepath.Base(path)
		if strings.HasPrefix(base, "mhrq_update_table_") || strings.HasPrefix(base, "mhrq_search_tokens_table_") || strings.HasPrefix(base, "mhrq_build_table_") {
			return nil
		}
		if strings.HasPrefix(base, "mhrq_comparison_config1_m_") || strings.HasPrefix(base, "mhrq_metrics_config1") {
			return ingestMHRQResultFile(path, out)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

func ingestMHRQResultFile(path string, out map[int]map[int][]mhrqSummaryRow) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		cols := strings.Split(line, "\t")
		if len(cols) < 2 {
			continue
		}
		phase := cols[0]
		row := mhrqSummaryRow{phase: phase}
		if len(cols) > 1 && cols[1] != "" {
			row.keywords, _ = strconv.Atoi(cols[1])
		}
		if len(cols) > 2 && cols[2] != "" {
			row.rangeWidth, _ = strconv.Atoi(cols[2])
		}
		if len(cols) > 4 && cols[4] != "" {
			row.buildNS, _ = strconv.ParseInt(cols[4], 10, 64)
		}
		if len(cols) > 5 && cols[5] != "" {
			row.queryNS, _ = strconv.ParseInt(cols[5], 10, 64)
		}
		if len(cols) > 6 && cols[6] != "" {
			row.updateNS, _ = strconv.ParseInt(cols[6], 10, 64)
		}
		if len(cols) > 7 && cols[7] != "" {
			row.storageByte, _ = strconv.Atoi(cols[7])
		}
		if len(cols) > 8 && cols[8] != "" {
			row.tokens, _ = strconv.Atoi(cols[8])
		}
		if len(cols) > 9 && cols[9] != "" {
			row.results, _ = strconv.Atoi(cols[9])
		}
		if row.keywords == 0 {
			continue
		}
		if _, ok := out[row.keywords]; !ok {
			out[row.keywords] = make(map[int][]mhrqSummaryRow)
		}
		switch phase {
		case "build_index", "build":
			out[row.keywords][0] = append(out[row.keywords][0], row)
		case "search":
			out[row.keywords][row.rangeWidth] = append(out[row.keywords][row.rangeWidth], row)
		case "update":
			out[row.keywords][-1] = append(out[row.keywords][-1], row)
		case "storage":
			out[row.keywords][-2] = append(out[row.keywords][-2], row)
		}
	}
	return s.Err()
}

func writeMHRQUpdateTable(path string, rows map[int]map[int][]mhrqSummaryRow) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	if _, err := fmt.Fprintln(w, "# of Inverted Index\tFB-DSSE(us)\tMHRQ(us)"); err != nil {
		return err
	}
	for _, k := range []int{5000, 10000, 15000, 20000, 25000} {
		avg := averageUpdateMicros(rows[k][-1])
		if _, err := fmt.Fprintf(w, "%d\t\t%.2f\n", k, avg); err != nil {
			return err
		}
	}
	return nil
}

func writeMHRQSearchTable(path string, rows map[int]map[int][]mhrqSummaryRow) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	if _, err := fmt.Fprintln(w, "Size of Range Query\tFB-DSSE(avg tokens)\tMHRQ(avg tokens)"); err != nil {
		return err
	}
	rangeWidths := []int{600, 1200, 1800, 2400, 3000, 3600, 4200, 4800}
	for _, rw := range rangeWidths {
		avg := averageTokens(rowsForRange(rows, rw))
		if _, err := fmt.Fprintf(w, "%.1f\t\t%.0f\n", float64(rw)/1000.0, avg); err != nil {
			return err
		}
	}
	return nil
}

func writeMHRQBuildTable(path string, rows map[int]map[int][]mhrqSummaryRow) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	if _, err := fmt.Fprintln(w, "# of Inverted Index\tFB-DSSE(ms)\tMHRQ(ms)"); err != nil {
		return err
	}
	keys := sortedSummaryKeys(rows)
	for _, k := range keys {
		avg := averageBuildMillis(rows[k][0])
		if _, err := fmt.Fprintf(w, "%d\t\t%.2f\n", k, avg); err != nil {
			return err
		}
	}
	return nil
}

func rowsForRange(rows map[int]map[int][]mhrqSummaryRow, rw int) []mhrqSummaryRow {
	var out []mhrqSummaryRow
	for _, m := range rows {
		out = append(out, m[rw]...)
	}
	return out
}

func sortedSummaryKeys(rows map[int]map[int][]mhrqSummaryRow) []int {
	keys := make([]int, 0, len(rows))
	for k := range rows {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	return keys
}

func sortedRangeWidths(rows map[int]map[int][]mhrqSummaryRow) []int {
	seen := map[int]struct{}{}
	for _, m := range rows {
		for rw := range m {
			if rw > 0 {
				seen[rw] = struct{}{}
			}
		}
	}
	out := make([]int, 0, len(seen))
	for rw := range seen {
		out = append(out, rw)
	}
	sort.Ints(out)
	return out
}

func averageUpdateMicros(rows []mhrqSummaryRow) float64 {
	if len(rows) == 0 {
		return 0
	}
	var sum int64
	for _, r := range rows {
		sum += r.durationNS
	}
	return float64(sum) / float64(len(rows)) / 1000.0
}

func averageBuildMillis(rows []mhrqSummaryRow) float64 {
	if len(rows) == 0 {
		return 0
	}
	var sum int64
	for _, r := range rows {
		sum += r.durationNS
	}
	return float64(sum) / float64(len(rows)) / 1e6
}

func averageTokens(rows []mhrqSummaryRow) float64 {
	if len(rows) == 0 {
		return 0
	}
	var sum int
	for _, r := range rows {
		sum += r.tokens
	}
	return float64(sum) / float64(len(rows))
}

func writeMHRQComparisonCSVLike(outPath string, lines []string) error {
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	for _, line := range lines {
		if _, err := fmt.Fprintln(w, line); err != nil {
			return err
		}
	}
	return nil
}
