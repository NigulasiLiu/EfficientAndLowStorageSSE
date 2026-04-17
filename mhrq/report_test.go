package mhrq

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRunReportFromCSVs(t *testing.T) {
	input := map[string]string{
		"mhrq": filepath.Join("results", "mhrq", "mhrq_metrics.csv"),
	}
	out := filepath.Join("results", "mhrq", "summary_all_schemes.csv")
	if err := RunReportFromCSVs(input, out); err != nil {
		t.Fatal(err)
	}
}

func TestRunThreeSchemeReport(t *testing.T) {
	input := map[string]string{}

	candidates := map[string]string{
		"fb_dsse": filepath.Join("results", "fb_dsse", "fb_dsse_metrics.csv"),
		"vh_rsse": filepath.Join("results", "vh_rsse", "vh_rsse_metrics.csv"),
		"mhrq":    filepath.Join("results", "mhrq", "mhrq_metrics.csv"),
	}

	for scheme, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			input[scheme] = path
		}
	}

	if len(input) == 0 {
		t.Skip("no scheme CSV files found to merge")
	}

	out := filepath.Join("results", "summary_three_schemes.csv")
	if err := RunReportFromCSVs(input, out); err != nil {
		t.Fatal(err)
	}
}
