package mhrq

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"math/big"
)

// Matrix is an N x N matrix of big integers.
type Matrix struct {
	N    int
	Data [][]*big.Int
}

func NewMatrix(n int) *Matrix {
	m := &Matrix{N: n, Data: make([][]*big.Int, n)}
	for i := range m.Data {
		m.Data[i] = make([]*big.Int, n)
		for j := range m.Data[i] {
			m.Data[i][j] = big.NewInt(0)
		}
	}
	return m
}

func IdentityMatrix(n int) *Matrix {
	m := NewMatrix(n)
	for i := 0; i < n; i++ {
		m.Data[i][i] = big.NewInt(1)
	}
	return m
}

func (m *Matrix) Clone() *Matrix {
	out := NewMatrix(m.N)
	for i := 0; i < m.N; i++ {
		for j := 0; j < m.N; j++ {
			out.Data[i][j] = new(big.Int).Set(m.Data[i][j])
		}
	}
	return out
}

func (m *Matrix) Mul(b *Matrix) *Matrix {
	if m.N != b.N {
		panic("matrix dimension mismatch")
	}
	out := NewMatrix(m.N)
	for i := 0; i < m.N; i++ {
		for j := 0; j < m.N; j++ {
			sum := big.NewInt(0)
			for k := 0; k < m.N; k++ {
				term := new(big.Int).Mul(m.Data[i][k], b.Data[k][j])
				sum.Add(sum, term)
			}
			out.Data[i][j] = sum
		}
	}
	return out
}

func (m *Matrix) Trace() *big.Int {
	s := big.NewInt(0)
	for i := 0; i < m.N; i++ {
		s.Add(s, m.Data[i][i])
	}
	return s
}

func RandBytes(n int) []byte {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return b
}

func HMACSHA256(key, msg []byte) []byte {
	mac := hmac.New(sha256.New, key)
	mac.Write(msg)
	return mac.Sum(nil)
}

func H1(key, msg []byte) []byte { return HMACSHA256(key, msg) }
func H2(key, msg []byte) []byte { return HMACSHA256(key, msg) }

func ZeroBytes(n int) []byte { return make([]byte, n) }

func IntToBytes(x int) []byte {
	return big.NewInt(int64(x)).Bytes()
}

func invertibleRandomMatrix(n int) *Matrix {
	return IdentityMatrix(n)
}
