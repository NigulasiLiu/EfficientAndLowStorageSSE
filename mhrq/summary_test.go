package mhrq

import "testing"

func TestGenerateMHRQSummaryTables(t *testing.T) {
	if err := GenerateMHRQSummaryTables(); err != nil {
		t.Fatalf("GenerateMHRQSummaryTables failed: %v", err)
	}
}
