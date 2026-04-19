package mhrq

import (
	"bufio"
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

func TestMHRQBuildSearchUpdateStorageTXT(t *testing.T) {
	outPath := filepath.Join("results", "mhrq", "mhrq_metrics_config1.txt")
	logPath := filepath.Join("results", "mhrq", "mhrq_experiment_progress.log")
	if err := testWriteMHRQMetricsTXT(outPath, logPath); err != nil {
		t.Fatal(err)
	}
}

func testWriteMHRQMetricsTXT(outPath, logPath string) error {
	SetMatrixGenMode(MatrixGenFast)
	outDir := filepath.Dir(outPath)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}

	logFile, err := os.Create(logPath)
	if err != nil {
		return err
	}
	defer logFile.Close()
	logW := bufio.NewWriter(logFile)
	defer logW.Flush()

	metricsFile, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer metricsFile.Close()
	metricsW := bufio.NewWriter(metricsFile)
	defer metricsW.Flush()

	if err := writeTXTHeader(metricsW); err != nil {
		return err
	}

	sizes := []int{5000, 10000, 15000, 20000, 25000}
	ranges := []int{600, 1200, 1800, 2400, 3000, 3600, 4200, 4800}

	_, _ = logW.WriteString(fmt.Sprintf("[%s] [MHRQ-Experiment] start: output=%s mode=%s\n", nowStamp(), outPath, CurrentMatrixGenMode()))
	for idx, n := range sizes {
		_, _ = logW.WriteString(fmt.Sprintf("[%s] [Build] dataset=%d (%d/%d) start\n", nowStamp(), n, idx+1, len(sizes)))
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
			if (i+1)%5000 == 0 || i+1 == n {
				_, _ = logW.WriteString(fmt.Sprintf("[%s] [Build] dataset=%d progress=%d/%d\n", nowStamp(), n, i+1, n))
			}
		}
		buildDuration := time.Since(startBuild).Nanoseconds()
		_, _ = metricsW.WriteString(fmt.Sprintf("build_index\t%d\t\t0\t%d\t\t\t%d\n", n, buildDuration, estimateStorageBytes(s)))
		_, _ = logW.WriteString(fmt.Sprintf("[%s] [Build] dataset=%d done duration_ns=%d\n", nowStamp(), n, buildDuration))

		_, _ = logW.WriteString(fmt.Sprintf("[%s] [Search] dataset=%d start (ranges=%d)\n", nowStamp(), n, len(ranges)))
		for rIdx, width := range ranges {
			// startSearch := time.Now()
			// res, err := s.Search("1", 1, width)
			// dur := time.Since(startSearch).Nanoseconds()

			// 修改为：
			res, clientTimeNs, serverTimeNs, err := s.Search("1", 1, width)
			dur := clientTimeNs + serverTimeNs // 或者保留原来的 time.Since
			if err != nil {
				return err
			}
			// 修改写入格式以包含这两个新时间
			_, _ = metricsW.WriteString(fmt.Sprintf("search\t%d\t%d\t0\t%d\t%d\t%d\t%d\t%d\t%d\n",
				n, width, dur, searchTokenCountByPseudoCode(), len(res), estimateStorageBytes(s),
				clientTimeNs, serverTimeNs)) // <--- 新增写入
			_, _ = logW.WriteString(fmt.Sprintf("[%s] [Search] dataset=%d range=%d (%d/%d) done duration_ns=%d tokens=%d\n", nowStamp(), n, width, rIdx+1, len(ranges), dur, searchTokenCountByPseudoCode()))
		}

		_, _ = logW.WriteString(fmt.Sprintf("[%s] [Update] dataset=%d start\n", nowStamp(), n))
		startUpdate := time.Now()
		_, err = s.Update(fmt.Sprintf("doc-%d-extra", n), "add", "heart", n)
		updateDuration := time.Since(startUpdate).Nanoseconds()
		if err != nil {
			return err
		}
		_, _ = metricsW.WriteString(fmt.Sprintf("update\t%d\t\t0\t%d\t\t\t%d\n", n, updateDuration, estimateStorageBytes(s)))
		_, _ = logW.WriteString(fmt.Sprintf("[%s] [Update] dataset=%d done duration_ns=%d\n", nowStamp(), n, updateDuration))

		_, _ = logW.WriteString(fmt.Sprintf("[%s] [Storage/Revoke] dataset=%d start\n", nowStamp(), n))
		startRevoke := time.Now()
		_, err = s.Revoke()
		revokeDuration := time.Since(startRevoke).Nanoseconds()
		if err != nil {
			return err
		}
		_, _ = metricsW.WriteString(fmt.Sprintf("storage\t%d\t\t0\t%d\t\t\t%d\n", n, revokeDuration, estimateStorageBytes(s)))
		_, _ = logW.WriteString(fmt.Sprintf("[%s] [Storage/Revoke] dataset=%d done duration_ns=%d\n", nowStamp(), n, revokeDuration))
		_ = logW.Flush()
		_ = metricsW.Flush()
	}
	_, _ = logW.WriteString(fmt.Sprintf("[%s] [MHRQ-Experiment] finished\n", nowStamp()))
	return nil
}

func TestMHRQComparisonDriver(t *testing.T) {
	if err := RunComparison(); err != nil {
		t.Fatal(err)
	}
}
