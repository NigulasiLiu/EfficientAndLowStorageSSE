package mhrq

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
)

type ExperimentRecord struct {
	Scheme      string
	Phase       string
	Keywords    int
	RangeWidth  int
	Iteration   int
	DurationNS  int64
	Tokens      int
	Results     int
	StorageByte int
}

func RunReportFromCSVs(inputFiles map[string]string, outputFile string) error {
	if err := os.MkdirAll(filepath.Dir(outputFile), 0o755); err != nil {
		return err
	}

	out, err := os.Create(outputFile)
	if err != nil {
		return err
	}
	defer out.Close()

	w := csv.NewWriter(out)
	defer w.Flush()
	if err := w.Write([]string{"scheme", "phase", "keywords", "range_width", "iteration", "duration_ns", "tokens", "results", "storage_bytes"}); err != nil {
		return err
	}

	schemes := make([]string, 0, len(inputFiles))
	for scheme := range inputFiles {
		schemes = append(schemes, scheme)
	}
	sort.Strings(schemes)

	for _, scheme := range schemes {
		path := inputFiles[scheme]
		recs, err := readExperimentCSV(path)
		if err != nil {
			return fmt.Errorf("read %s: %w", path, err)
		}
		for _, r := range recs {
			if err := w.Write([]string{
				scheme,
				r.Phase,
				strconv.Itoa(r.Keywords),
				strconv.Itoa(r.RangeWidth),
				strconv.Itoa(r.Iteration),
				strconv.FormatInt(r.DurationNS, 10),
				strconv.Itoa(r.Tokens),
				strconv.Itoa(r.Results),
				strconv.Itoa(r.StorageByte),
			}); err != nil {
				return err
			}
		}
	}
	return nil
}

func readExperimentCSV(path string) ([]ExperimentRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	out := make([]ExperimentRecord, 0, len(records))
	for i, row := range records {
		if i == 0 || len(row) < 8 {
			continue
		}
		kw, _ := strconv.Atoi(row[1])
		rw := 0
		if row[2] != "" {
			rw, _ = strconv.Atoi(row[2])
		}
		iter, _ := strconv.Atoi(row[3])
		dur, _ := strconv.ParseInt(row[4], 10, 64)
		toks, _ := strconv.Atoi(row[5])
		res, _ := strconv.Atoi(row[6])
		st, _ := strconv.Atoi(row[7])
		out = append(out, ExperimentRecord{
			Scheme:      filepath.Base(path),
			Phase:       row[0],
			Keywords:    kw,
			RangeWidth:  rw,
			Iteration:   iter,
			DurationNS:  dur,
			Tokens:      toks,
			Results:     res,
			StorageByte: st,
		})
	}
	return out, nil
}
