package mhrq

import "math/big"

func bitVector(x, n int) []*big.Int {
	v := make([]*big.Int, 2*n)
	for i := 0; i < n; i++ {
		weight := new(big.Int).Lsh(big.NewInt(1), uint(n-1-i))
		v[2*i] = new(big.Int).Set(weight)
		v[2*i+1] = new(big.Int).Set(weight)
		if (x>>(n-1-i))&1 == 0 {
			v[2*i+1].Neg(v[2*i+1])
		}
	}
	return v
}

func outerProduct(v []*big.Int) *Matrix {
	m := NewMatrix(len(v))
	for i := 0; i < len(v); i++ {
		for j := 0; j < len(v); j++ {
			m.Data[i][j] = new(big.Int).Mul(v[i], v[j])
		}
	}
	return m
}

func embedExpansions(base *Matrix, r1, r2 *big.Int, n int) *Matrix {
	dim := 2*n + 2
	out := NewMatrix(dim)
	for i := 0; i < 2*n; i++ {
		for j := 0; j < 2*n; j++ {
			out.Data[i][j] = new(big.Int).Set(base.Data[i][j])
		}
	}
	for i := 0; i < dim; i++ {
		for j := 2 * n; j < dim; j++ {
			if out.Data[i][j] == nil {
				out.Data[i][j] = big.NewInt(0)
			}
			if out.Data[j][i] == nil {
				out.Data[j][i] = big.NewInt(0)
			}
		}
	}
	out.Data[2*n][2*n] = new(big.Int).Set(r1)
	out.Data[2*n+1][2*n+1] = new(big.Int).Set(r2)
	return out
}

func CRQ_Enc(x, n int, M1, M2 *Matrix) *Matrix {
	p := bitVector(x, n)
	P := outerProduct(p)
	Pbar := embedExpansions(P, randBigInt(128), randBigInt(128), n)
	return M1.Mul(Pbar).Mul(M2)
}

func CRQ_TokenGen(a, b, n int, M1, M2 *Matrix) *Matrix {
	p := bitVector(a, n)
	q := bitVector(b, n)
	Q := outerProduct(q)
	for i := 0; i < len(p) && i < Q.N; i++ {
		Q.Data[i][i].Add(Q.Data[i][i], p[i])
	}
	Qbar := embedExpansions(Q, randBigInt(128), randBigInt(128), n)
	return M2.Mul(Qbar).Mul(M1)
}

func CRQ_Query(PHat, QHat *Matrix) bool {
	return RangeTrace(PHat, QHat).Sign() < 0
}

func RangeTrace(P, Q *Matrix) *big.Int { return P.Mul(Q).Trace() }
