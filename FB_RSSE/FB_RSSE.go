package FB_RSSE

import (
	"EfficientAndLowStroageSSE/config"
	"crypto/sha256"
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"runtime"
	"strconv"
	"strings"
)

type Data struct {
	BigIntValue *big.Int
	ByteValue   []byte
}

type Counter struct {
	tokens []byte
	c      int
}

type bitmap struct {
	bs *big.Int
	c  int
}

// SystemParameters 系统参数
type SystemParameters struct {
	lambda        int
	K             []byte              // 系统密钥
	d             []int               // 查询范围
	H1            func([]byte) []byte // 哈希函数 H1
	H2            func([]byte) []byte // 哈希函数 H2
	EDB           map[string]Data     // 用于存储加密数据，类型改为 map[string]Data
	CT            map[string]Counter  // 计数器
	DB            map[string]bitmap   // 计数器
	KeywordToSK   map[string][]byte   // 每个关键词对应的密钥
	BsLength      int                 // Bitmap 长度
	n             *big.Int            // 公共参数 n
	LocalTree     map[string][]int64  // BRC tree
	localTreeCode map[string]string   // BRC tree
	TreeHeight    int
}

// input 是输入数据，返回伪随机的输出
func (sp *SystemParameters) PRF(input []byte) []byte {
	// 生成当前时间戳作为唯一标识符
	//timestamp := time.Now().UnixNano()

	// 将系统密钥、时间戳和输入数据结合
	combined := append(sp.K, input...) // 将密钥和输入数据结合
	//combined = append(combined, []byte(fmt.Sprintf("%d", timestamp))...) // 添加时间戳（确保每次调用唯一）

	// 使用 H1 哈希函数生成伪随机输出
	return sp.H1(combined)
}

// Setup 初始化系统参数
func Setup(L int) *SystemParameters {
	lambda := 256
	// 生成随机密钥
	K := make([]byte, 16)
	rand.Read(K)
	// 定义 H1 和 H2
	H1 := func(data []byte) []byte {
		hash := sha256.Sum256(data)
		return hash[:]
	}
	H2 := func(data []byte) []byte {
		hash := sha256.Sum256(data)
		return hash[:]
	}

	// 初始化 n 为 2^L
	n := new(big.Int).Exp(big.NewInt(2), big.NewInt(int64(L)), nil)

	// 初始化 EDB 和其他结构
	return &SystemParameters{
		lambda:      lambda,
		K:           K,
		H1:          H1,
		H2:          H2,
		EDB:         make(map[string]Data),
		DB:          make(map[string]bitmap),
		CT:          make(map[string]Counter),
		KeywordToSK: make(map[string][]byte),
		BsLength:    L,
		d:           config.Range,
		n:           n, // 初始化 n
	}
}

// BuildIndex 构建倒排索引
func (sp *SystemParameters) BuildDB(invertedIndex map[string][]int) (int, error) {
	// 创建一个大整数表示位图
	for keyword, docIDs := range invertedIndex {
		code := sp.localTreeCode[keyword]
		// 从代码的长度开始，逐步截取每次减少一位直到只剩一个字符
		for i := len(code); i >= 0; i-- {
			tempCode := code[:i]
			tempCode = tempCode + strings.Repeat("*", len(code)-len(tempCode))

			info := getOrDefault(sp.DB, tempCode, bitmap{
				c:  -1,
				bs: big.NewInt(0),
			})
			info.c++
			// 遍历 docIDs，将每个文档ID映射到位图的对应位置
			for _, docID := range docIDs {
				// 将对应的位设置为 1，使用 big.Int 的 SetBit 方法
				info.bs.SetBit(info.bs, docID, 1)
			}
			sp.DB[tempCode] = info
			//// 打印 keyword 和对应的 info.bs 的最低 50 位
			//fmt.Printf("tempCode: %s: ", tempCode)
			//PrintLowestBits(info.bs, 50)
		}
	}
	return len(sp.DB), nil
}

// BuildIndex 构建倒排索引
func (sp *SystemParameters) BuildIndex(invertedIndex map[string][]int, sortedKeywords []string) error {
	sp.TreeHeight = int(math.Ceil(math.Log2(float64(len(invertedIndex)))))
	// 构建 LocalTree
	sp.buildLocalTree(sortedKeywords)
	_, err := sp.BuildDB(invertedIndex)

	if err != nil {
		return err
	}
	for tempCode, tempBitmap := range sp.DB {
		//加密索引
		K_w := sp.PRF([]byte(tempCode))
		ST_c, _ := sp.GenerateRandom()
		ST_cplus1, _ := sp.GenerateRandom()
		c := tempBitmap.c
		// 将 int 转换为 string
		c_str := strconv.Itoa(c)
		// 将 string 转换为 []byte
		c_str_Array := []byte(c_str)
		//将DB中的计数器值和新计算的令牌值存在CT
		sp.CT[tempCode] = Counter{c: c, tokens: ST_cplus1}
		sk := sp.H1(append(K_w, c_str_Array...))
		UT_cplus1 := sp.H1(append(K_w, ST_cplus1...))
		encBitmap := sp.Enc(new(big.Int).SetBytes(sk), tempBitmap.bs)

		//fmt.Printf("Build Index tempCode: %s: ", tempCode)
		//PrintLowestBits(encBitmap, 50)

		C_ST, _ := XOR(UT_cplus1, ST_c)
		//fmt.Printf("C_ST: %x\n", C_ST)
		sp.EDB[string(UT_cplus1)] = Data{
			BigIntValue: encBitmap,
			ByteValue:   C_ST,
		}

	}

	return nil
}

func (sp *SystemParameters) BuildDBMock1(invertedIndex map[string][]int) (int, error) {
	// 创建一个大整数表示位图
	for keyword, docIDs := range invertedIndex {
		code := sp.localTreeCode[keyword]
		// 从代码的长度开始，逐步截取每次减少一位直到只剩一个字符
		for i := len(code); i >= 0; i-- {
			tempCode := code[:i]
			tempCode = tempCode + strings.Repeat("*", len(code)-len(tempCode))

			info := getOrDefault(sp.DB, tempCode, bitmap{
				c:  -1,
				bs: big.NewInt(0),
			})
			info.c++
			// 遍历 docIDs，将每个文档ID映射到位图的对应位置
			for _, docID := range docIDs {
				// 将对应的位设置为 1，使用 big.Int 的 SetBit 方法
				//docID = 1000
				info.bs.SetBit(info.bs, docID, 1)
			}
			sp.DB[tempCode] = info
			//// 打印 keyword 和对应的 info.bs 的最低 50 位
			//fmt.Printf("tempCode: %s: ", tempCode)
			//PrintLowestBits(info.bs, 50)
		}
	}
	return len(sp.DB), nil
}

// BuildIndex 构建倒排索引
func (sp *SystemParameters) BuildIndexMock1(invertedIndex map[string][]int, sortedKeywords []string) error {
	sp.TreeHeight = int(math.Ceil(math.Log2(float64(len(invertedIndex)))))
	// 构建 LocalTree
	sp.buildLocalTree(sortedKeywords)
	_, err := sp.BuildDBMock(invertedIndex)

	if err != nil {
		return err
	}
	for tempCode, tempBitmap := range sp.DB {
		//加密索引
		K_w := sp.PRF([]byte(tempCode))
		ST_c, _ := sp.GenerateRandom()
		ST_cplus1, _ := sp.GenerateRandom()
		c := tempBitmap.c
		// 将 int 转换为 string
		c_str := strconv.Itoa(c)
		// 将 string 转换为 []byte
		c_str_Array := []byte(c_str)
		//将DB中的计数器值和新计算的令牌值存在CT
		sp.CT[tempCode] = Counter{c: c, tokens: ST_cplus1}
		sk := sp.H1(append(K_w, c_str_Array...))
		UT_cplus1 := sp.H1(append(K_w, ST_cplus1...))
		encBitmap := sp.Enc(new(big.Int).SetBytes(sk), tempBitmap.bs)

		//fmt.Printf("Build Index tempCode: %s: ", tempCode)
		//PrintLowestBits(encBitmap, 50)

		C_ST, _ := XOR(UT_cplus1, ST_c)
		//fmt.Printf("C_ST: %x\n", C_ST)
		sp.EDB[string(UT_cplus1)] = Data{
			BigIntValue: encBitmap,
			ByteValue:   C_ST,
		}

	}

	return nil
}

// BuildDBMock 生成临时位图数据（不存储到sp.DB，直接返回用于加密）
// 返回值：临时映射表（tempCode -> bitmap），用于BuildIndexMock处理后立即销毁
func (sp *SystemParameters) BuildDBMock(invertedIndex map[string][]int) (map[string]bitmap, error) {
	// 临时存储生成的位图，仅在BuildIndexMock中使用后即销毁
	tempDB := make(map[string]bitmap)

	for keyword, docIDs := range invertedIndex {
		code, exists := sp.localTreeCode[keyword]
		if !exists {
			return nil, fmt.Errorf("keyword %s not found in localTreeCode", keyword)
		}

		// 生成所有前缀编码（与原逻辑一致）
		for i := len(code); i >= 0; i-- {
			tempCode := code[:i]
			tempCode = tempCode + strings.Repeat("*", len(code)-len(tempCode))

			// 从临时映射中获取或初始化bitmap（不使用sp.DB）
			info := getOrDefault(tempDB, tempCode, bitmap{
				c:  -1,
				bs: big.NewInt(0), // 临时big.Int，仅在当前循环中使用
			})
			info.c++

			// 在位图中设置文档ID对应的位（核心临时操作）
			for _, docID := range docIDs {
				info.bs.SetBit(info.bs, docID%1e6, 1) //取模以减小索引构建需要耗费的内存
			}
			tempDB[tempCode] = info
		}

		// 每处理完一个关键词，手动触发一次GC，回收临时big.Int
		runtime.GC()
	}

	return tempDB, nil
}

// BuildIndexMock 边生成EDB边清除临时数据，不依赖全局DB存储
func (sp *SystemParameters) BuildIndexMock(invertedIndex map[string][]int, sortedKeywords []string) error {
	sp.TreeHeight = int(math.Ceil(math.Log2(float64(len(invertedIndex)))))
	sp.buildLocalTree(sortedKeywords)

	// 1. 生成临时位图数据（不存入sp.DB，仅在内存中临时持有）
	tempDB, err := sp.BuildDBMock(invertedIndex)
	if err != nil {
		return err
	}
	// 延迟清理临时DB（确保加密完成后立即释放）
	defer func() {
		tempDB = nil // 解除引用
		runtime.GC() // 强制回收临时DB中的所有big.Int
	}()

	// 2. 遍历临时位图，加密后写入EDB，同时清理临时变量
	for tempCode, tempBitmap := range tempDB {
		// 加密索引所需的临时变量（局部定义，迭代后自动销毁）
		K_w := sp.PRF([]byte(tempCode))
		ST_c, _ := sp.GenerateRandom()
		ST_cplus1, _ := sp.GenerateRandom()
		c := tempBitmap.c

		// 临时转换计数器（局部变量，用完即销毁）
		c_str := strconv.Itoa(c)
		c_str_Array := []byte(c_str)

		// 更新CT（仅保留必要的计数器和令牌，无冗余数据）
		sp.CT[tempCode] = Counter{c: c, tokens: ST_cplus1}

		// 生成加密密钥（临时big.Int，加密后立即丢弃）
		skBytes := sp.H1(append(K_w, c_str_Array...))
		sk := new(big.Int).SetBytes(skBytes) // 局部临时变量

		// 加密位图（使用临时DB中的bitmap，加密后不再引用）
		encBitmap := sp.Enc(sk, tempBitmap.bs)

		// 生成并存储到EDB
		UT_cplus1 := sp.H1(append(K_w, ST_cplus1...))
		C_ST, _ := XOR(UT_cplus1, ST_c)
		sp.EDB[string(UT_cplus1)] = Data{
			BigIntValue: encBitmap,
			ByteValue:   C_ST,
		}

		// 清除当前迭代的临时大对象（关键：主动释放big.Int引用）
		sk = nil
		tempBitmap.bs = nil // 解除对位图的引用
	}

	// 最终清理：确保所有临时数据被回收
	runtime.GC()
	return nil
}

// buildLocalTreeFromClusters 构建 LocalTree
func (sp *SystemParameters) buildLocalTree(sortedKeywords []string) {
	localTreeCode := make(map[string]string)
	for _, keyword := range sortedKeywords {
		intkey, _ := strconv.Atoi(keyword)
		tempKey := fmt.Sprintf("%0*b", sp.TreeHeight+1, intkey) // 二进制表示
		localTreeCode[keyword] = tempKey
		//fmt.Fprintf(file, "keyword == %s, localTreeCode: %s\n", keyword, tempKey)
	}
	//// 保存树的最终内容
	//err = encoder.Encode(localTreeCode)
	//if err != nil {
	//	fmt.Println("Error encoding to JSON:", err)
	//}
	// 保存树和分区信息
	//sp.LocalTree = localTree
	sp.localTreeCode = localTreeCode
}

// TPath generates the set of node encodings (w) from the root to the keyword.
// This is equivalent to generating all prefixes of the keyword's BRC/BPC code.
func (sp *SystemParameters) TPath(keyword string) ([]string, error) {
	code, exists := sp.localTreeCode[keyword]
	if !exists {
		return nil, fmt.Errorf("keyword %s not found in localTreeCode", keyword)
	}

	var pt []string
	// 从代码的长度开始，逐步截取每次减少一位直到只剩一个字符
	// 这里的逻辑需要匹配所用的树结构。在BuildDB中的逻辑，是生成所有前缀并用 '*' 填充。
	// 但在动态更新中，通常只对路径上的节点更新。为了匹配BuildDB/BuildIndex逻辑，
	// 我们生成带 '*' 的全长编码。
	for i := len(code); i >= 0; i-- {
		tempCode := code[:i]
		// 填充星号到原始代码长度
		tempCode = tempCode + strings.Repeat("*", len(code)-len(tempCode))
		if i == 0 {
			// 跳过全星号（根节点）的特殊情况，如果需要更新根节点，可以保留。
			// 为了安全，我们通常从 i=1 开始或根据具体协议来。这里保留了 i=0 的逻辑。
		}

		// 只有在 i > 0 时才添加，因为 i=0 会导致一个全星号的键，可能不需要。
		if i > 0 {
			pt = append(pt, tempCode)
		}
	}
	// 注意：这里的路径生成可能与实际的树结构实现有所不同。
	// 如果使用 BRC/BPC 结构，通常只需要路径上的非叶子节点。
	return pt, nil
}

// Update implements the dynamic update algorithm as specified in the image.
// Parameters:
// keyword: The keyword to be updated.
// bs: The new bitmap (bs) for the keyword (represented as *big.Int).
func (sp *SystemParameters) UpdateBigInt(keyword string, bs *big.Int) error {
	// 1. 获取路径上的所有编码 PT
	PT, err := sp.TPath(keyword)
	if err != nil {
		return err
	}

	// 2. 遍历 PT (for w ∈ PT)
	for _, w := range PT {
		// 4: Kw||K'w ← FK(w), (STc, c) ← CT[w]
		// 简化 F_K(w) 的实现：使用 PRF 生成 Kw，使用 H1(Kw) 作为 K'w
		// 在实际系统中，这应该是一个定义明确的 PRF 且输出长度足够分割。
		Kw := sp.PRF([]byte(w))
		// K'w 假设为 Kw 的一个哈希值，或者 F_K(w) 输出的后半部分
		// 这里简化为 K'w = H2(Kw)
		Kw_prime := sp.H2(Kw)

		// (STc, c) ← CT[w]
		info := getOrDefault(sp.CT, w, Counter{c: -1, tokens: []byte{}})
		ST_c := info.tokens
		c := info.c

		// 5-7: if (STc, c) = ⊥ then c ← −1, STc ← {0, 1}λ end if
		if c == -1 {
			// c 已经设置为 -1
			ST_c, _ = sp.GenerateRandom() // 生成随机值
		}

		// 8: STc+1 ← {0, 1}λ
		ST_cplus1, _ := sp.GenerateRandom()
		c_plus1 := c + 1

		// 9: CT[w] ← (STc+1, c + 1)
		sp.CT[w] = Counter{tokens: ST_cplus1, c: c_plus1}

		// 将 int (c+1) 转换为 []byte
		c_plus1_str := strconv.Itoa(c_plus1)
		c_plus1_Array := []byte(c_plus1_str)

		// 10: UTc+1 ← H1(Kw, STc+1)
		UT_cplus1 := sp.H1(append(Kw, ST_cplus1...))

		// 11: CSTc ← H2(Kw, STc+1) ⊕ STc
		// 注意：协议图中使用 H2(Kw, STc+1)，而 BuildIndex 中使用了 UTc+1 (即 H1(Kw, STc+1))。
		// 为遵循图片算法，我们使用 H2。
		H2_output := sp.H2(append(Kw, ST_cplus1...))
		C_STc, err := XOR(H2_output, ST_c)
		if err != nil {
			return fmt.Errorf("XOR error for C_STc: %v", err)
		}

		// 12: skc+1 ← H3(K'w, c + 1)
		// 假设 H3 等于 H1
		sk_cplus1_bytes := sp.H1(append(Kw_prime, c_plus1_Array...))
		sk_cplus1 := new(big.Int).SetBytes(sk_cplus1_bytes)

		// 13: ec+1 ← Enc(skc+1, bs, n)
		e_cplus1 := sp.Enc(sk_cplus1, bs)
		e_cplus1 = nil
		// 14: Send (UTc+1,(ec+1, CSTc)) to the server. (即更新 EDB)
		// Server: 17: Set EDB[UTc+1] ← (ec+1, CSTc)
		sp.EDB[string(UT_cplus1)] = Data{
			BigIntValue: e_cplus1, // ec+1
			ByteValue:   C_STc,    // CSTc
		}
	}

	return nil
}

// Update 更新FB_RSSE的索引，文档ID改为int类型
func (sp *SystemParameters) Update(keyword string, bs int) error {
	// 1. 获取路径上的所有编码 PT
	PT, err := sp.TPath(keyword)
	if err != nil {
		return err
	}

	// 2. 遍历 PT (for w ∈ PT)
	for _, w := range PT {
		// 4: Kw||K'w ← FK(w), (STc, c) ← CT[w]
		// 简化 F_K(w) 的实现：使用 PRF 生成 Kw，使用 H1(Kw) 作为 K'w
		// 在实际系统中，这应该是一个定义明确的 PRF 且输出长度足够分割。
		Kw := sp.PRF([]byte(w))
		// K'w 假设为 Kw 的一个哈希值，或者 F_K(w) 输出的后半部分
		// 这里简化为 K'w = H2(Kw)
		Kw_prime := sp.H2(Kw)

		// (STc, c) ← CT[w]
		info := getOrDefault(sp.CT, w, Counter{c: -1, tokens: []byte{}})
		ST_c := info.tokens
		c := info.c

		// 5-7: if (STc, c) = ⊥ then c ← −1, STc ← {0, 1}λ end if
		if c == -1 {
			// c 已经设置为 -1
			ST_c, _ = sp.GenerateRandom() // 生成随机值
		}

		// 8: STc+1 ← {0, 1}λ
		ST_cplus1, _ := sp.GenerateRandom()
		c_plus1 := c + 1

		// 9: CT[w] ← (STc+1, c + 1)
		sp.CT[w] = Counter{tokens: ST_cplus1, c: c_plus1}

		// 将 int (c+1) 转换为 []byte
		c_plus1_str := strconv.Itoa(c_plus1)
		c_plus1_Array := []byte(c_plus1_str)

		// 10: UTc+1 ← H1(Kw, STc+1)
		UT_cplus1 := sp.H1(append(Kw, ST_cplus1...))

		// 11: CSTc ← H2(Kw, STc+1) ⊕ STc
		// 注意：协议图中使用 H2(Kw, STc+1)，而 BuildIndex 中使用了 UTc+1 (即 H1(Kw, STc+1))。
		// 为遵循图片算法，我们使用 H2。
		H2_output := sp.H2(append(Kw, ST_cplus1...))
		C_STc, err := XOR(H2_output, ST_c)
		if err != nil {
			return fmt.Errorf("XOR error for C_STc: %v", err)
		}

		// 12: skc+1 ← H3(K'w, c + 1)
		// 假设 H3 等于 H1
		sk_cplus1_bytes := sp.H1(append(Kw_prime, c_plus1_Array...))
		sk_cplus1 := new(big.Int).SetBytes(sk_cplus1_bytes)

		// 13: ec+1 ← Enc(skc+1, bs, n)
		bigIntBs := new(big.Int).SetInt64(int64(bs))
		e_cplus1 := sp.Enc(sk_cplus1, bigIntBs)

		// 14: Send (UTc+1,(ec+1, CSTc)) to the server. (即更新 EDB)
		// Server: 17: Set EDB[UTc+1] ← (ec+1, CSTc)
		sp.EDB[string(UT_cplus1)] = Data{
			BigIntValue: e_cplus1, // ec+1
			ByteValue:   C_STc,    // CSTc
		}
	}
	return nil
}
func (sp *SystemParameters) GenToken(queryRange [2]string, sortedKeywords []string) ([][]byte, [][]byte, []int, error) {
	targetValue, _ := sp.getBRC(queryRange, sortedKeywords)
	//fmt.Println("BRC:", targetValue)
	var K_w_set [][]byte
	var ST_set [][]byte
	var c_set []int
	for _, tempCode := range targetValue {
		//fmt.Println("Dealing with:", tempCode)
		//加密索引
		K_w := sp.PRF([]byte(tempCode))
		info := getOrDefault(sp.CT, tempCode, Counter{c: -1, tokens: []byte{}})
		if info.c == -1 {
			return nil, nil, []int{-1}, nil
		}
		K_w_set = append(K_w_set, K_w)
		ST_set = append(ST_set, info.tokens)
		c_set = append(c_set, info.c)
	}

	return K_w_set, ST_set, c_set, nil
}
func (sp *SystemParameters) ServerSearch(K_w_set [][]byte, ST_set [][]byte, c_set []int) (*big.Int, error) {
	// 初始化 Sum 为 0
	var Sum = big.NewInt(0)
	// 用于存储每个 Sum_e
	var Sum_e = big.NewInt(0)
	for index, K_w_i := range K_w_set {
		// 初始化 Sum_e 为 0
		Sum_e.SetInt64(0)
		// 从 c_set 的最后一个元素开始遍历
		ST_j := ST_set[index]
		for j := c_set[index]; j >= 0; j-- {
			UT_j := sp.H1(append(K_w_i, ST_j...))
			// 以十六进制格式打印所有三个字节数组，在一行中
			//fmt.Printf("K_w_i in hex: %x, ST_j in hex: %x, UT_j in hex: %x\n", K_w_i, ST_j, UT_j)
			data := sp.EDB[string(UT_j)]
			if data.ByteValue == nil {
				break
			}
			Sum_e = sp.Add(Sum_e, data.BigIntValue)
			delete(sp.EDB, string(UT_j))
			ST_j, _ = XOR(UT_j, data.ByteValue)
			//fmt.Printf("ST_j in hex: %x, data.ByteValue in hex: %x\n", ST_j, data.ByteValue)
		}

		// 将当前 Sum_e 加到 Sum 中
		Sum = sp.Add(Sum, Sum_e)
	}
	// 返回最终的 Sum
	return Sum, nil
}
func (sp *SystemParameters) LocalParse(K_w_set [][]byte, c_set []int, Sum *big.Int) (*big.Int, error) {
	// 用于存储每个 Sum_e
	var Sum_sk = big.NewInt(0)
	var sk_i = big.NewInt(0)
	for i, K_w_i := range K_w_set {
		for j := c_set[i]; j >= 0; j-- {
			// 将 int 转换为 string
			c_str := strconv.Itoa(j)
			// 将 string 转换为 []byte
			c_str_Array := []byte(c_str)
			sk_i_bytes := sp.H1(append(K_w_i, c_str_Array...))
			sk_i.SetBytes(sk_i_bytes)
			Sum_sk = sp.Add(Sum_sk, sk_i)
		}
		// 打印每个 Sum_sk 对应的位图
		//PrintBitmap(Sum_sk, 50) // 打印 Sum_sk 的最低 30 位
	}
	//Sum_sk.Set()
	return sp.Dec(Sum_sk, Sum), nil
}

// searchTree 在本地树中查找关键词的位置
func (sp *SystemParameters) searchTree(sortedKeywords []string, queryValue string, findLarger bool) (string, error) {
	queryValueIndex := indexOf(sortedKeywords, queryValue)
	if queryValueIndex != -1 {
		return sp.localTreeCode[sortedKeywords[queryValueIndex]], nil
	}
	queryRangeInt, _ := strconv.Atoi(queryValue)
	// 将 localCluster[0] 转换为 []int
	sortedKeywordsInt := make([]int, len(sortedKeywords))
	for i, v := range sortedKeywords {
		sortedKeywordsInt[i], _ = strconv.Atoi(v)
	}
	// 找到比 queryRange[0] 大且差距最小的值
	tempIndex := binarySearchClosest(sortedKeywordsInt, queryRangeInt, findLarger)

	return sortedKeywords[tempIndex], nil
}
func (sp *SystemParameters) getBRC(queryRange [2]string, sortedKeywords []string) ([]string, error) {
	newQueryLeft, _ := sp.searchTree(sortedKeywords, queryRange[0], true)
	newQueryRight, _ := sp.searchTree(sortedKeywords, queryRange[1], false)
	if newQueryLeft == newQueryRight {
		return []string{newQueryLeft}, nil
	}
	a, _ := strconv.ParseInt(newQueryLeft, 2, 64)
	b, _ := strconv.ParseInt(newQueryRight, 2, 64)
	rangeInt := make([]int, b-a+1)
	for i := range rangeInt {
		// 将 int64 转换为 int，确保类型匹配
		rangeInt[i] = int(a + int64(i))
	}
	cover, err := sp.preCover(rangeInt)
	if err != nil {
		return nil, err
	}
	// 最后返回的 resultSet
	return cover, nil
}

// GetBPCValueMap 获取BPC值的映射
func (sp *SystemParameters) GetBPCValueMap(R []int, bits int) (map[int][]int, error) {
	currentShiftCounts := make(map[int][]int) // 当前轮的右移次数
	// 当前轮的int集合
	currentSet := make(map[int]struct{})

	// 将R的int值放入currentSet
	for _, value := range R {
		currentSet[value] = struct{}{}
	}

	iteration := 0
	// 终止条件：currentSet大小为1或达到bits次迭代
	for len(currentSet) > 1 && iteration < bits {
		mapShifted := make(map[int][]int)

		// 构建父子节点映射关系
		for value := range currentSet {
			// 模拟右移操作（右移一位）
			parentValue := value >> 1

			// 将右移后的结果放入map
			mapShifted[parentValue] = append(mapShifted[parentValue], value)
		}

		// 清空currentSet，为下一轮迭代准备
		currentSet = make(map[int]struct{})

		// 遍历mapShifted，决定是将右移后的父节点保留，还是将子节点添加到resultSet
		for parentValue, values := range mapShifted {
			if len(values) > 1 {
				// 如果当前父节点有多个子节点，则保留父节点
				currentSet[parentValue] = struct{}{}
			} else {
				// 将当前值添加到当前轮的结果集
				currentShiftCounts[iteration] = append(currentShiftCounts[iteration], values...)
			}
		}

		iteration++ // 增加迭代次数
	}

	// 将最后的currentSet中的值加入到结果集中
	if len(currentSet) > 0 {
		currentShiftCounts[iteration] = append(currentShiftCounts[iteration], mapKeys(currentSet)...)
	}

	return currentShiftCounts, nil
}

// 辅助函数：从map中提取出keys
func mapKeys(m map[int]struct{}) []int {
	keys := make([]int, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	return keys
}

// convertMapToPrefixString 将 BPC 结果转换为二进制字符串列表
func (sp *SystemParameters) convertMapToPrefixString(resultMap map[int][]int, bits int) []string {
	var binaryStrings []string

	// 遍历每次迭代的结果
	for iteration, values := range resultMap {
		for _, value := range values {
			// 将 int 转换为二进制字符串
			binaryString := fmt.Sprintf("%b", value)

			// 计算有效长度和前缀位数
			effectiveLength := bits - iteration
			leadingZeros := max(effectiveLength-len(binaryString), 0)

			// 构建完整的二进制字符串（前导0 + 二进制值 + '*'）
			sb := strings.Builder{}
			for i := 0; i < leadingZeros; i++ {
				sb.WriteByte('0')
			}
			sb.WriteString(binaryString)

			// 添加 '*' 标识符
			for i := 0; i < iteration; i++ {
				sb.WriteByte('*')
			}

			// 添加到结果列表
			binaryStrings = append(binaryStrings, sb.String())
		}
	}

	return binaryStrings
}

// convertToOnlyPrefix 将 BPC 结果转换为只有前缀的二进制字符串列表
func (sp *SystemParameters) convertToOnlyPrefix(resultMap map[int][]int, bits int) []string {
	var binaryStrings []string

	// 遍历每次迭代的结果
	for iteration, values := range resultMap {
		for _, value := range values {
			// 将 int 转换为二进制字符串
			binaryString := fmt.Sprintf("%b", value)

			// 计算有效长度和前缀位数
			effectiveLength := bits - iteration
			leadingZeros := max(effectiveLength-len(binaryString), 0)

			// 构建完整的二进制字符串（前导0 + 二进制值）
			sb := strings.Builder{}
			for i := 0; i < leadingZeros; i++ {
				sb.WriteByte('0')
			}
			sb.WriteString(binaryString)

			// 添加到结果列表
			binaryStrings = append(binaryStrings, sb.String())
		}
	}

	return binaryStrings
}

// preCover 实现前缀编码生成
func (sp *SystemParameters) preCover(R []int) ([]string, error) {
	// 获取BPC结果（包括分组）
	resultMap, err := sp.GetBPCValueMap(R, sp.TreeHeight+1)
	if err != nil {
		return nil, fmt.Errorf("error generating BPC value map: %v", err)
	}

	// 返回BPC前缀字符串
	return sp.convertMapToPrefixString(resultMap, sp.TreeHeight+1), nil
}

// Enc 加密方法
func (sp *SystemParameters) Enc(sk, m *big.Int) *big.Int {
	// Enc(sk, m) = (sk + m) % n
	result := new(big.Int).Add(sk, m)
	result.Mod(result, sp.n)
	//return result
	return m
}

// Dec 解密方法
func (sp *SystemParameters) Dec(sk, e *big.Int) *big.Int {
	// Dec(sk, e) = (e - sk + n) % n
	result := new(big.Int).Sub(e, sk)
	result.Add(result, sp.n)
	result.Mod(result, sp.n)
	//return result
	return e
}

// Add 加法同态方法
func (sp *SystemParameters) Add(e1, e2 *big.Int) *big.Int {
	// Add(e1, e2) = (e1 + e2) % n
	result := new(big.Int).Add(e1, e2)
	result.Mod(result, sp.n)
	return result
}

// GenerateRandomBytesAndConvertToBigInt 生成长度为 lambda 位的随机字节数组，并转换为 BigInt
func (sp *SystemParameters) GenerateRandom() ([]byte, error) {
	// 计算字节数组的长度
	byteLength := (sp.lambda + 7) / 8 // lambda 位对应的字节数，确保向上取整

	// 生成随机字节数组
	randomBytes := make([]byte, byteLength)
	_, err := rand.Read(randomBytes)
	if err != nil {
		return nil, err // 生成随机字节失败
	}

	// 将字节数组转换为 BigInt
	//randomBigInt := new(big.Int).SetBytes(randomBytes)

	return randomBytes, nil
}

// XOR 函数：对两个字节数组按位异或
func XOR(b1, b2 []byte) ([]byte, error) {
	// 如果两个字节数组的长度不同，返回错误
	if len(b1) != len(b2) {
		return nil, fmt.Errorf("byte arrays must have the same length")
	}

	// 创建一个新的字节数组来存储结果
	result := make([]byte, len(b1))

	// 遍历字节数组并执行 XOR 操作
	for i := 0; i < len(b1); i++ {
		result[i] = b1[i] ^ b2[i]
	}

	return result, nil
}
func indexOf(slice []string, value string) int {
	for i, v := range slice {
		if v == value {
			return i
		}
	}
	return -1 // 未找到返回 -1
}
func binarySearchClosest(slice []int, value int, findLarger bool) int {
	low, high := 0, len(slice)-1
	closest := -1

	for low <= high {
		mid := (low + high) / 2
		if slice[mid] == value {
			return mid
		}
		if findLarger {
			if slice[mid] > value {
				closest = mid
				high = mid - 1
			} else {
				low = mid + 1
			}
		} else {
			if slice[mid] < value {
				closest = mid
				low = mid + 1
			} else {
				high = mid - 1
			}
		}
	}
	if closest < 0 {
		return 0
	}
	return closest
}

// getOrDefault 使用泛型处理任意类型的 map
func getOrDefault[K comparable, V any](m map[K]V, key K, defaultValue V) V {
	// 尝试从 map 中获取值
	if value, exists := m[key]; exists {
		// 如果 key 存在，返回对应的值
		return value
	}
	// 如果 key 不存在，返回默认值
	return defaultValue
}

// BuildIndex 构建倒排索引
func (sp *SystemParameters) BuildIndex_dynamic(invertedIndex map[string][]int, sortedKeywords []string) error {
	sp.TreeHeight = int(math.Ceil(math.Log2(float64(len(invertedIndex)))))
	// 构建 LocalTree
	sp.buildLocalTree(sortedKeywords)
	// 创建一个大整数表示位图
	for keyword, docIDs := range invertedIndex {
		code := sp.localTreeCode[keyword]
		// 从代码的长度开始，逐步截取每次减少一位直到只剩一个字符
		for i := len(code); i > 0; i-- {
			tempCode := code[:i]

			info := getOrDefault(sp.CT, tempCode,
				Counter{
					tokens: []byte{}, // 默认 tokens 是一个包含一个字节数组的切片
					c:      -1,       // 默认计数器为 0
				})
			if info.c == -1 {
				info.tokens, _ = sp.GenerateRandom()
			}

			//加密索引
			K_w := sp.PRF([]byte(tempCode))
			ST_cplus1, _ := sp.GenerateRandom()
			sp.CT[tempCode] = Counter{
				tokens: ST_cplus1,  // 更新 tokens 为 ST_cplus1
				c:      info.c + 1, // 更新 c 为 info.c + 1
			}

			//记录，用于搜索
			sp.KeywordToSK[tempCode] = ST_cplus1
			UT_cplus1 := sp.H1(append(K_w, ST_cplus1...))
			C_ST, _ := XOR(sp.H1(append(K_w, ST_cplus1...)), ST_cplus1)

			// 加密位图
			bitmap := big.NewInt(0)
			// 遍历 docIDs，将每个文档ID映射到位图的对应位置
			for _, docID := range docIDs {
				// 将对应的位设置为 1，使用 big.Int 的 SetBit 方法
				bitmap.SetBit(bitmap, docID+1, 1)
			}
			// 将 int 转换为 string
			c_str := strconv.Itoa(info.c + 1)
			// 将 string 转换为 []byte
			c_str_Array := []byte(c_str)
			encBitmap := sp.Enc(new(big.Int).SetBytes(sp.H1(append(K_w, c_str_Array...))), bitmap)
			sp.EDB[string(UT_cplus1)] = Data{
				BigIntValue: encBitmap,
				ByteValue:   C_ST,
			}

		}
	}

	return nil
}

// PrintBitmap 输出大整数中所有为 1 的位的位置，并将它们在一行打印出来
func PrintBitmap(bitmap *big.Int, maxBits int) {
	// 用于存储所有为1的位的索引
	var positions []int

	// 遍历所有位
	for i := 0; i < maxBits; i++ {
		if bitmap.Bit(i) == 1 {
			// 将为 1 的位的索引添加到 positions 切片中
			positions = append(positions, i)
		}
	}

	// 打印所有为1的位的位置，格式化为一行
	fmt.Printf("Positions with 1 bits: [%s]\n", fmt.Sprint(positions))
}

// PrintLowestBits 打印 big.Int 对应的最低若干位二进制字符串
func PrintLowestBits(encBitmap *big.Int, maxBits int) {
	// 获取 big.Int 的二进制表示
	bits := encBitmap.Text(2) // 获取 big.Int 的二进制字符串

	// 如果位数超过 maxBits，截取最低 maxBits 位
	if len(bits) > maxBits {
		bits = bits[len(bits)-maxBits:] // 截取最后 maxBits 位
	} else {
		// 如果位数少于 maxBits，补充前导零
		bits = fmt.Sprintf("%0*s", maxBits, bits)
	}

	// 打印二进制字符串的最低若干位
	fmt.Printf("info.bs (lowest %d bits): %s\n", maxBits, bits)
}
