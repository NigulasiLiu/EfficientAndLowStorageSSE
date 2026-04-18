package mhrq

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestRunAllMHRQExperiments runs the MHRQ benchmark pipeline end-to-end.
// It generates the raw CSV and then produces a combined summary CSV.
func TestRunAllMHRQExperiments(t *testing.T) {
	outDir := filepath.Join("results", "mhrq")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		t.Fatal(err)
	}

	t.Run("raw_experiment_txt", func(t *testing.T) {
		if err := runMHRQExperimentTXT(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("summary_txt", func(t *testing.T) {
		input := map[string]string{
			"mhrq": filepath.Join(outDir, "mhrq_metrics_config1.txt"),
		}
		output := filepath.Join(outDir, "summary_all_schemes.txt")
		if err := RunReportFromCSVs(input, output); err != nil {
			t.Fatal(err)
		}
	})

	fmt.Println("MHRQ experiment pipeline completed successfully")
}

func runMHRQExperimentTXT() error {
	return testWriteMHRQMetricsTXT(
		filepath.Join("results", "mhrq", "mhrq_metrics_config1.txt"),
		filepath.Join("results", "mhrq", "mhrq_experiment_progress.log"),
	)
}
