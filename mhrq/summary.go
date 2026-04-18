package mhrq

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
)

func writeMHRQSummaryTables(resultsDir string) error {
	if err := os.MkdirAll(resultsDir, 0o755); err != nil {
		return err
	}

	// These tables are intended to be pasted directly into the rightmost column
	// of the existing DSSE_CTRQ vs FB-DSSE result tables.
	files := map[string]string{
		"update": filepath.Join(resultsDir, "mhrq_update_table_config1.txt"),
		"search": filepath.Join(resultsDir, "mhrq_search_tokens_table_config1.txt"),
		"build":  filepath.Join(resultsDir, "mhrq_build_table_config1.txt"),
	}

	updateF, err := os.Create(files["update"])
	if err != nil {
		return err
	}
	defer updateF.Close()
	updateW := bufio.NewWriter(updateF)
	defer updateW.Flush()
	_, _ = updateW.WriteString("# of Inverted Index\tFB-DSSE(us)\tMHRQ(us)\n")
	_, _ = updateW.WriteString("5\t88.42\t\n")
	_, _ = updateW.WriteString("10\t99.37\t\n")
	_, _ = updateW.WriteString("15\t126.87\t\n")
	_, _ = updateW.WriteString("20\t198.7\t\n")
	_, _ = updateW.WriteString("25\t294.55\t\n")

	searchF, err := os.Create(files["search"])
	if err != nil {
		return err
	}
	defer searchF.Close()
	searchW := bufio.NewWriter(searchF)
	defer searchW.Flush()
	_, _ = searchW.WriteString("Size of Range Query\tFB-DSSE(avg tokens)\tMHRQ(avg tokens)\n")
	_, _ = searchW.WriteString("1.2\t10.52\t2\n")
	_, _ = searchW.WriteString("1.8\t10.82\t2\n")
	_, _ = searchW.WriteString("2.4\t11.74\t2\n")
	_, _ = searchW.WriteString("3.0\t11.76\t2\n")
	_, _ = searchW.WriteString("3.6\t12.4\t2\n")
	_, _ = searchW.WriteString("4.2\t12.14\t2\n")
	_, _ = searchW.WriteString("4.8\t14.3\t2\n")

	buildF, err := os.Create(files["build"])
	if err != nil {
		return err
	}
	defer buildF.Close()
	buildW := bufio.NewWriter(buildF)
	defer buildW.Flush()
	_, _ = buildW.WriteString("# of Inverted Index\tFB-DSSE(ms)\tMHRQ(ms)\n")
	_, _ = buildW.WriteString("10\t313831.02214\t\n")
	_, _ = buildW.WriteString("15\t474153.57608\t\n")
	_, _ = buildW.WriteString("20\t621638.73714\t\n")
	_, _ = buildW.WriteString("25\t822250.99303\t\n")

	return nil
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
