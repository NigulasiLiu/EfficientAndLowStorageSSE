package mhrq

import "testing"

func TestMatrixModeSwitch(t *testing.T) {
	SetMatrixGenMode(MatrixGenFast)
	if CurrentMatrixGenMode() != MatrixGenFast {
		t.Fatalf("expected fast mode")
	}
	SetMatrixGenMode(MatrixGenStrict)
	if CurrentMatrixGenMode() != MatrixGenStrict {
		t.Fatalf("expected strict mode")
	}
	SetMatrixGenMode("unknown")
	if CurrentMatrixGenMode() != MatrixGenFast {
		t.Fatalf("unknown mode should fallback to fast")
	}
}
