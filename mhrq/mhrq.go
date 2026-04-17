package mhrq

import (
	"fmt"
	"math/big"
	"sync"
)

// Scheme holds the state for MHRQ.
type Scheme struct {
	mu     sync.RWMutex
	n      int
	setup  *SetupParams
	edb    map[string][]*Ciphertext
	st     map[string][]byte
	revCnt uint64
	kprf   []byte
}

func Setup(lambda, n int) (*Scheme, error) {
	_ = lambda
	m1 := invertibleRandomMatrix(2*n + 2)
	m2 := invertibleRandomMatrix(2*n + 2)
	kprf := RandBytes(32)
	kse := RandBytes(32)
	return &Scheme{
		n:     n,
		setup: &SetupParams{N: n, M1: m1, M2: m2, KSE: kse, KPRF: kprf},
		edb:   make(map[string][]*Ciphertext),
		st:    make(map[string][]byte),
		kprf:  kprf,
	}, nil
}

func (s *Scheme) dprf(ts uint64, keyword string) []byte {
	return H1(s.kprf, []byte(fmt.Sprintf("%s|%d", keyword, ts)))[:16]
}

func (s *Scheme) keywordToken(keyword string) []byte {
	return H2(s.kprf, []byte(keyword+"|1"))
}

func zero32() []byte { return ZeroBytes(32) }

func (s *Scheme) Update(id, op, w string, x int) (*Ciphertext, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	stc := s.dprf(uint64(len(s.edb[w])+1), w)
	ad := H2(s.kprf, []byte(w+"|"+id+"|"+op))
	encP := CRQ_Enc(x, s.n, s.setup.M1, s.setup.M2)
	node := &Ciphertext{
		ID: id, Keyword: w, Op: op, Value: x,
		Timestamp: uint64(len(s.edb[w]) + 1),
		ST:        stc, PrevST: append([]byte{}, s.st[w]...), Address: ad,
		Delta: zero32(), Token: s.keywordToken(w), EncryptedP: encP, Searchable: true,
	}
	s.edb[w] = append(s.edb[w], node)
	s.st[w] = stc
	return node, nil
}

func (s *Scheme) Revoke() (*RevokeToken, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	oldKey := append([]byte{}, s.kprf...)
	newKey := RandBytes(32)
	delta := H1(newKey, oldKey)
	m3 := invertibleRandomMatrix(2*s.n + 2)
	m4 := invertibleRandomMatrix(2*s.n + 2)
	mPrime := m3.Mul(s.setup.M1)
	mDPrime := s.setup.M2.Mul(m4)

	s.kprf = newKey
	s.setup.KPRF = newKey
	s.revCnt++

	for _, list := range s.edb {
		for _, ct := range list {
			ct.EncryptedP = mPrime.Mul(ct.EncryptedP).Mul(mDPrime)
			ct.Token = s.keywordToken(ct.Keyword)
		}
	}

	// Insert an explicit revoke node to drive chain traversal.
	for keyword := range s.edb {
		node := &Ciphertext{
			ID:         "revoke-" + fmt.Sprintf("%d", s.revCnt),
			Keyword:    keyword,
			Op:         "revoke",
			Timestamp:  uint64(len(s.edb[keyword]) + 1),
			ST:         append([]byte{}, s.st[keyword]...),
			PrevST:     append([]byte{}, s.st[keyword]...),
			Delta:      append([]byte{}, delta...),
			Token:      s.keywordToken(keyword),
			IsRevoke:   true,
			Searchable: true,
		}
		s.edb[keyword] = append(s.edb[keyword], node)
		break
	}

	return &RevokeToken{NewKPRF: newKey, MPrime: mPrime, MDPrime: mDPrime, Delta: delta, M3: m3, M4: m4}, nil
}

func (s *Scheme) Search(w string, a, b int) ([]SearchResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	qhat := CRQ_TokenGen(a, b, s.n, s.setup.M1, s.setup.M2)
	results := make([]SearchResult, 0)
	currentToken := append([]byte{}, s.keywordToken(w)...)

	for i := len(s.edb[w]) - 1; i >= 0; i-- {
		ct := s.edb[w][i]
		if ct == nil {
			continue
		}
		if ct.IsRevoke || !isZeroBytes(ct.Delta) {
			currentToken = keyUpdate(ct.Delta, currentToken)
			continue
		}
		if ct.EncryptedP == nil {
			continue
		}
		trace := RangeTrace(ct.EncryptedP, qhat)
		if trace.Sign() < 0 {
			results = append(results, SearchResult{ID: ct.ID, Keyword: ct.Keyword, Value: ct.Value, Matched: true, Trace: trace, CipherID: ct.ID})
		}
	}
	_ = currentToken
	return results, nil
}

func keyUpdate(delta, token []byte) []byte {
	if len(delta) == 0 {
		return append([]byte{}, token...)
	}
	if len(token) == 0 {
		return append([]byte{}, delta...)
	}
	out := make([]byte, len(token))
	for i := range token {
		out[i] = token[i] ^ delta[i%len(delta)]
	}
	return out
}

func isZeroBytes(b []byte) bool {
	for _, v := range b {
		if v != 0 {
			return false
		}
	}
	return true
}

func (s *Scheme) ExportState() map[string]any {
	return map[string]any{"keywords": len(s.edb), "revocations": s.revCnt}
}

func (s *Scheme) ECUpdateToken() []byte {
	sum := H1(s.kprf, []byte("ec-update"))
	return new(big.Int).SetBytes(sum).Bytes()
}
