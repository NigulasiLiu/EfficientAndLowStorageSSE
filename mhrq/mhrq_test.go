package mhrq

import "testing"

func TestFullFlow(t *testing.T) {
	s, err := Setup(128, 8)
	if err != nil {
		t.Fatalf("setup failed: %v", err)
	}

	if _, err := s.Update("1", "add", "heart", 12); err != nil {
		t.Fatalf("update1 failed: %v", err)
	}
	if _, err := s.Update("2", "add", "heart", 18); err != nil {
		t.Fatalf("update2 failed: %v", err)
	}
	if _, err := s.Update("3", "add", "heart", 25); err != nil {
		t.Fatalf("update3 failed: %v", err)
	}

	res, err := s.Search("heart", 10, 20)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(res) == 0 {
		t.Fatalf("expected search results before revoke")
	}

	if _, err := s.Revoke(); err != nil {
		t.Fatalf("revoke failed: %v", err)
	}

	if _, err := s.Update("4", "add", "heart", 15); err != nil {
		t.Fatalf("post-revoke update failed: %v", err)
	}

	res2, err := s.Search("heart", 10, 20)
	if err != nil {
		t.Fatalf("search after revoke failed: %v", err)
	}
	if len(res2) == 0 {
		t.Fatalf("expected search results after revoke")
	}
}
