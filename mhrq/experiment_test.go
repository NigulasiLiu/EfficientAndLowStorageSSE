package mhrq

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

func BenchmarkMHRQBasic(b *testing.B) {
	s, err := Setup(128, 32)
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < b.N; i++ {
		if _, err := s.Update(fmt.Sprintf("id-%d", i), "add", "heart", i%1000); err != nil {
			b.Fatal(err)
		}
	}
}

func TestMHRQBuildSearchUpdateStorageCSV(t *testing.T) {
	if err := testWriteMHRQMetrics(filepath.Join("results", "mhrq", "mhrq_metrics.csv")); err != nil {
		t.Fatal(err)
	}
}

func testWriteMHRQMetrics(outPath string) error {
	outDir := filepath.Dir(outPath)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}

	sizes := []int{5000, 10000, 15000, 20000, 25000}
	ranges := []int{600, 1200, 1800, 2400, 3000, 3600, 4200, 4800}
	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	_ = w.Write([]string{"phase", "keywords", "range_width", "iteration", "duration_ns", "tokens", "results", "storage_bytes"})

	for _, n := range sizes {
		s, err := Setup(128, 32)
		if err != nil {
			return err
		}

		startBuild := time.Now()
		for i := 0; i < n; i++ {
			_, err := s.Update(fmt.Sprintf("doc-%d", i+1), "add", strconv.Itoa(i+1), i)
			if err != nil {
				return err
			}
		}
		buildDuration := time.Since(startBuild).Nanoseconds()
		_ = w.Write([]string{"build_index", strconv.Itoa(n), "", "0", strconv.FormatInt(buildDuration, 10), "", "", strconv.Itoa(estimateStorageBytes(s))})

		for _, width := range ranges {
			startSearch := time.Now()
			res, err := s.Search("1", 1, width)
			dur := time.Since(startSearch).Nanoseconds()
			if err != nil {
				return err
			}
			_ = w.Write([]string{"search", strconv.Itoa(n), strconv.Itoa(width), "0", strconv.FormatInt(dur, 10), strconv.Itoa(2), strconv.Itoa(len(res)), strconv.Itoa(estimateStorageBytes(s))})
		}

		startUpdate := time.Now()
		_, err = s.Update(fmt.Sprintf("doc-%d-extra", n), "add", "heart", n)
		updateDuration := time.Since(startUpdate).Nanoseconds()
		if err != nil {
			return err
		}
		_ = w.Write([]string{"update", strconv.Itoa(n), "", "0", strconv.FormatInt(updateDuration, 10), "", "", strconv.Itoa(estimateStorageBytes(s))})

		startRevoke := time.Now()
		_, err = s.Revoke()
		revokeDuration := time.Since(startRevoke).Nanoseconds()
		if err != nil {
			return err
		}
		_ = w.Write([]string{"storage", strconv.Itoa(n), "", "0", strconv.FormatInt(revokeDuration, 10), "", "", strconv.Itoa(estimateStorageBytes(s))})
	}
	return nil
}

func TestMHRQComparisonDriver(t *testing.T) {
	if err := RunComparison(); err != nil {
		t.Fatal(err)
	}
}
