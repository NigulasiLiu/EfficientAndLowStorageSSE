package OurScheme

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"math"
	"math/big"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// OurScheme 系统参数
type OurScheme struct {
	L             int                 // 每个分区允许的最大大小
	Key           []byte              // 系统密钥
	H1            func([]byte) []byte // 哈希函数 H1
	H2            func([]byte) []byte // 哈希函数 H2
	EDB           map[string][]byte   // 加密数据库
	LocalTree     map[string][]int64  // 更改为存储整数的 map
	ClusterFlist  [][]int             // 分区文件列表
	ClusterKlist  [][]string          // 分区关键词列表
	KeywordToSK   map[string][]byte   // 每个关键词对应的 OTP 密钥
	BsLength      int                 // Bitmap 长度
	LocalPosition [2]int              // 查询范围对应的分区位置
	Flags         []string            // 标记需要查询的边界（左边界 "l"，右边界 "r"）
	FlagEmpty     []string            // 标记查询结果是否为空

}

// Setup 初始化系统参数
func Setup(L int) *OurScheme {
	// 生成随机密钥
	key := make([]byte, 16)
	rand.Read(key)

	// 定义 H1 和 H2
	H1 := func(data []byte) []byte {
		hash := sha256.Sum256(data)
		return hash[:]
	}
	H2 := func(data []byte) []byte {
		hash := sha256.Sum256(data)
		return hash[:]
	}

	// 初始化 EDB 和其他结构
	return &OurScheme{
		L:            L,
		Key:          key,
		H1:           H1,
		H2:           H2,
		EDB:          make(map[string][]byte),
		LocalTree:    make(map[string][]int64),
		ClusterFlist: [][]int{},
		ClusterKlist: [][]string{},
		KeywordToSK:  make(map[string][]byte),
		BsLength:     L,
	}
}

// ---------------------------------------------------------
// 核心模块 2: 索引构建 (BuildIndex)
// ---------------------------------------------------------
func (sp *OurScheme) BuildIndex(invertedIndex map[string][]int, keywords []string) error {
	// 3. 构建 VH-RSSE 分区结构 (Partitioning)
	currentGroup := []int{}      // 当前分区的文件 ID
	currentKlist := []string{}   // 当前分区的关键词
	clusterFlist := [][]int{}    // 所有分区的文件 ID
	clusterKlist := [][]string{} // 所有分区的关键词
	//clusterVolume := []int{}     // 分区的文件数量

	// 遍历每个关键词
	for i, keyword := range keywords {
		postings := invertedIndex[keyword]

		// 检查是否可以加入当前分区
		if len(currentGroup)+len(postings) < sp.L {
			currentGroup = append(currentGroup, postings...)
			currentKlist = append(currentKlist, keyword)

			// 加密并存储
			sp.encryptAndStore(keyword, currentGroup)

			// 如果是最后一个关键词，保存当前分区
			if i == len(keywords)-1 {
				clusterFlist = append(clusterFlist, append([]int{}, currentGroup...))
				clusterKlist = append(clusterKlist, append([]string{}, currentKlist...))
				//clusterVolume = append(clusterVolume, len(currentGroup))
			}
		} else {
			// 保存当前分区
			clusterFlist = append(clusterFlist, append([]int{}, currentGroup...))
			clusterKlist = append(clusterKlist, append([]string{}, currentKlist...))
			//clusterVolume = append(clusterVolume, len(currentGroup))

			// 初始化新分区
			currentGroup = append([]int{}, postings...)
			currentKlist = append([]string{}, keyword)

			// 加密并存储
			sp.encryptAndStore(keyword, currentGroup)

			// 如果是最后一个关键词，保存新分区
			if i == len(keywords)-1 {
				clusterFlist = append(clusterFlist, append([]int{}, currentGroup...))
				clusterKlist = append(clusterKlist, append([]string{}, currentKlist...))
				//clusterVolume = append(clusterVolume, len(currentGroup))
			}
		}
	}

	// 保存分区结果
	sp.ClusterFlist = clusterFlist
	//fmt.Println("ClusterFlist:", sp.ClusterFlist)
	sp.ClusterKlist = clusterKlist
	//fmt.Println("ClusterKlist:", sp.ClusterKlist)

	// 构建 LocalTree
	sp.buildLocalTree(clusterKlist)

	return nil
}

// buildLocalTreeFromClusters 构建 LocalTree
func (sp *OurScheme) buildLocalTree(clusterKlist [][]string) {
	genList := [][]string{}
	for _, klist := range clusterKlist {
		if len(klist) > 0 {
			genList = append(genList, []string{klist[0], klist[len(klist)-1]}) // 每个分区的第一个和最后一个关键词
		}
	}

	// 填充到最近的 2 的幂次
	clusterHeight := int(math.Ceil(math.Log2(float64(len(genList)))))
	//fmt.Printf("clusterHeight: %d\n", clusterHeight)
	padding := genList[len(genList)-1][1]
	for len(genList) < int(math.Pow(2, float64(clusterHeight))) {
		genList = append(genList, []string{padding, padding})
	}

	//// 创建输出文件
	//file, err := os.Create("C:\\Users\\Admin\\Desktop\\GoPros\\EfficientAndLowStroageSSE\\dataset\\tree_new.txt")
	//if err != nil {
	//	fmt.Println("Error creating file:", err)
	//	return
	//}
	//defer file.Close()
	//// 写入文件的JSON编码器
	//encoder := json.NewEncoder(file)
	//encoder.SetIndent("", "  ") // 设置格式化输出

	// 构建树
	localTree := make(map[string][]int64)
	for i := clusterHeight; i >= 0; i-- {
		for j := 0; j < int(math.Pow(2, float64(i))); j++ {
			tempKey := fmt.Sprintf("%0*b", i+1, j) // 二进制表示
			if i == clusterHeight {
				// 将关键词转换为整数，存储在 LocalTree 中
				leftv, _ := strconv.ParseInt(genList[j][0], 10, 64)
				rightv, _ := strconv.ParseInt(genList[j][1], 10, 64)
				localTree[tempKey] = append(localTree[tempKey], leftv)
				localTree[tempKey] = append(localTree[tempKey], rightv)
				// 输出到文件
				//fmt.Fprintf(file, "i == %d, tempKey = %s: left = %d, right = %d\n", i, tempKey, leftv, rightv)
			} else {
				//localTree[tempKey][0] = localTree[tempKey+"0"][0]
				//localTree[tempKey][1] = localTree[tempKey+"1"][1]
				localTree[tempKey] = append(localTree[tempKey], localTree[tempKey+"0"][0])
				localTree[tempKey] = append(localTree[tempKey], localTree[tempKey+"1"][1])
			}
		}
	}
	//// 保存树的最终内容
	//err = encoder.Encode(localTree)
	//if err != nil {
	//	fmt.Println("Error encoding to JSON:", err)
	//}
	// 保存树和分区信息
	sp.LocalTree = localTree
	//sp.LocalTree["volume"] = int64(len(clusterVolume)) // 保存分区文件数量
}

// encryptAndStore 加密并存储
func (sp *OurScheme) encryptAndStore(keyword string, postings []int) {
	// 生成 Bitmap
	//log.Printf("group len: %d, sp.L: %d", len(postings), sp.L)
	bitmap := sp.generateBitmap(postings)
	//log.Printf("bitmap_string: %s", bitmap)
	// 生成 OTP 密钥
	otpKey := sp.H1([]byte(keyword))

	// 加密
	encryptedBitmap := xorBytesWithPadding(bitmap, otpKey, sp.L)

	// 存储到 EDB
	hashedKey := hex.EncodeToString(sp.H1([]byte(keyword)))
	sp.KeywordToSK[hashedKey] = otpKey
	sp.EDB[hashedKey] = encryptedBitmap
}

// --------------------------
// OurScheme.Update 方法修改（适配新入参+红黑树插入）
// --------------------------
// Update 方法：根据关键词定位分区，将docID数组中每个ID插入红黑树
// 新签名：func (sp *OurScheme) Update(w string, docID []*big.Int) error
func (sp *OurScheme) Update(w string, docID []*big.Int) error {
	// 步骤1：通过本地树搜索定位关键词所在的分区索引P_K
	p, err := sp.searchTree(w)
	if err != nil {
		return fmt.Errorf("定位关键词[%s]分区失败: %v", w, err)
	}
	P_K := p

	// 步骤2：校验分区索引有效性
	if P_K < 0 || P_K >= len(sp.ClusterKlist) {
		return fmt.Errorf("关键词[%s]分区索引P_K=%d无效，超出范围[0,%d]", w, P_K, len(sp.ClusterKlist)-1)
	}

	// 步骤3：定位对应的文件分区P_F（与P_K索引一致）
	P_F := P_K
	if P_F < 0 || P_F >= len(sp.ClusterFlist) {
		return fmt.Errorf("文件分区索引P_F=%d无效，超出范围[0,%d]", P_F, len(sp.ClusterFlist)-1)
	}

	// 步骤4：初始化红黑树，将docID数组中每个ID插入红黑树
	// 注意：红黑树当前存储int类型，需将big.Int转换为int（若docID超出int范围，需修改红黑树Value类型为*big.Int）
	rbtree := NewRBTree()
	for _, id := range docID {
		// 将big.Int转换为int（假设docID值在int范围内，若超出需调整红黑树Value类型）
		idInt := int(id.Int64())
		rbtree.Insert(idInt)
	}

	// 步骤5：将红黑树关联到文件分区（示例：存储根节点值到文件列表，可根据实际需求扩展）
	sp.modifyFunction(&sp.ClusterFlist[P_F], rbtree)

	return nil
}

// modifyFunction 适配修改：接收红黑树，将树中数据关联到文件列表
func (sp *OurScheme) modifyFunction(fileList *[]int, rbtree *RBTree) {
	// 示例：将红黑树中所有节点值遍历添加到文件列表（验证插入效果）
	var traverse func(node *RBNode)
	traverse = func(node *RBNode) {
		if node == rbtree.NIL {
			return
		}
		traverse(node.Left)
		*fileList = append(*fileList, node.Value)
		traverse(node.Right)
	}
	traverse(rbtree.Root)
}

// 红黑树节点定义（保持原有逻辑）
type RBNode struct {
	Value  int
	Color  bool // true: 红色, false: 黑色
	Left   *RBNode
	Right  *RBNode
	Parent *RBNode
}

// 红黑树结构定义（保持原有逻辑）
type RBTree struct {
	Root *RBNode
	NIL  *RBNode // 哨兵节点（简化边界处理）
}

// 初始化红黑树（保持原有逻辑）
func NewRBTree() *RBTree {
	nilNode := &RBNode{Color: false} // 哨兵节点为黑色
	return &RBTree{
		NIL:  nilNode,
		Root: nilNode,
	}
}

// 红黑树插入函数（保持原有逻辑）
func (t *RBTree) Insert(value int) {
	newNode := &RBNode{
		Value: value,
		Color: true, // 新节点默认为红色
		Left:  t.NIL,
		Right: t.NIL,
	}

	// 标准BST插入
	var parent *RBNode = nil
	current := t.Root
	for current != t.NIL {
		parent = current
		if newNode.Value < current.Value {
			current = current.Left
		} else {
			current = current.Right
		}
	}
	newNode.Parent = parent
	if parent == nil {
		t.Root = newNode // 树为空时，新节点为根
	} else if newNode.Value < parent.Value {
		parent.Left = newNode
	} else {
		parent.Right = newNode
	}

	// 根节点特殊处理（设为黑色）
	if newNode.Parent == nil {
		newNode.Color = false
		return
	}
	// 父节点为根节点时无需调整
	if newNode.Parent.Parent == nil {
		return
	}

	// 插入后修复红黑树性质
	t.fixInsert(newNode)
}

// 插入后修复红黑树（保持原有逻辑）
func (t *RBTree) fixInsert(z *RBNode) {
	for z.Parent.Color {
		if z.Parent == z.Parent.Parent.Left {
			y := z.Parent.Parent.Right // 叔节点
			if y.Color {
				// 情况1：叔节点为红色，只需变色
				z.Parent.Color = false
				y.Color = false
				z.Parent.Parent.Color = true
				z = z.Parent.Parent
			} else {
				if z == z.Parent.Right {
					// 情况2：叔节点为黑色，当前节点为右孩子，先左旋
					z = z.Parent
					t.leftRotate(z)
				}
				// 情况3：叔节点为黑色，当前节点为左孩子，右旋+变色
				z.Parent.Color = false
				z.Parent.Parent.Color = true
				t.rightRotate(z.Parent.Parent)
			}
		} else {
			// 镜像情况（父节点为右孩子）
			y := z.Parent.Parent.Left
			if y.Color {
				z.Parent.Color = false
				y.Color = false
				z.Parent.Parent.Color = true
				z = z.Parent.Parent
			} else {
				if z == z.Parent.Left {
					z = z.Parent
					t.rightRotate(z)
				}
				z.Parent.Color = false
				z.Parent.Parent.Color = true
				t.leftRotate(z.Parent.Parent)
			}
		}
		if z == t.Root {
			break
		}
	}
	t.Root.Color = false // 确保根节点为黑色
}

// 左旋操作（保持原有逻辑）
func (t *RBTree) leftRotate(x *RBNode) {
	y := x.Right
	x.Right = y.Left
	if y.Left != t.NIL {
		y.Left.Parent = x
	}
	y.Parent = x.Parent
	if x.Parent == nil {
		t.Root = y
	} else if x == x.Parent.Left {
		x.Parent.Left = y
	} else {
		x.Parent.Right = y
	}
	y.Left = x
	x.Parent = y
}

// 右旋操作（保持原有逻辑）
func (t *RBTree) rightRotate(y *RBNode) {
	x := y.Left
	y.Left = x.Right
	if x.Right != t.NIL {
		x.Right.Parent = y
	}
	x.Parent = y.Parent
	if y.Parent == nil {
		t.Root = x
	} else if y == y.Parent.Right {
		y.Parent.Right = x
	} else {
		y.Parent.Left = x
	}
	x.Right = y
	y.Parent = x
}

// ensureDirExists 确保目录存在（保持原有逻辑）
func ensureDirExists(dir string) error {
	_, err := os.Stat(dir)
	if os.IsNotExist(err) {
		return os.MkdirAll(dir, 0755)
	}
	return err
}

// saveResult 保存测试结果（保持原有框架）
func saveResult(dir string, count, L int, data map[string]interface{}) error {
	// 实际实现可使用encoding/json将data序列化到文件（示例框架）
	return nil
}

func (sp *OurScheme) GenToken(queryRange [2]string) ([]string, error) {
	sp.FlagEmpty = []string{}
	sp.Flags = []string{}
	sp.LocalPosition = [2]int{}

	// 通过搜索树确定查询范围对应的分区位置
	p1, err := sp.searchTree(queryRange[0])
	if err != nil {
		return nil, fmt.Errorf("无法解析查询范围的起始位置：%v", err)
	}
	p2, err := sp.searchTree(queryRange[1])
	if err != nil {
		return nil, fmt.Errorf("无法解析查询范围的结束位置：%v", err)
	}
	if p1 > p2+1 {
		return []string{}, nil
	}
	sp.LocalPosition = [2]int{p1, p2}

	// 打印分区范围
	//log.Printf("Query range: %v, LocalPosition: %v", queryRange, sp.LocalPosition)

	// 获取查询范围对应的分区关键词列表
	localCluster := sp.ClusterKlist[p1 : p2+1]
	//log.Printf("Local cluster for query range: %v", localCluster)

	// 如果查询范围的起点和终点完全包含在分区中，则不需要额外的服务器查询
	if queryRange[0] == localCluster[0][0] && queryRange[1] == localCluster[len(localCluster)-1][len(localCluster[len(localCluster)-1])-1] {
		//log.Printf("Query range fully covered by local cluster, no tokens required.")
		return []string{}, nil
	}

	// 需要查询服务器的 token
	serverTokens := []string{}
	if queryRange[0] != localCluster[0][0] { // 如果查询的“左边界”值不是某个区间的“左边界”值
		tempIndex := indexOf(localCluster[0], queryRange[0]) - 1 //找到该关键字的位置，然后取该关键字左边的关键字的下标，因为这是闭区间，所以选择更小的值作为开区间边界
		if tempIndex < 0 {
			// 将 queryRange[0] 转换为 int
			queryRangeInt, err := strconv.Atoi(queryRange[0])
			if err != nil {
				log.Fatalf("Failed to convert queryRange[0] to int: %v", err)
			}

			// 将 localCluster[0] 转换为 []int
			localClusterInt := make([]int, len(localCluster[0]))
			for i, v := range localCluster[0] {
				localClusterInt[i], err = strconv.Atoi(v)
				if err != nil {
					log.Fatalf("Failed to convert localCluster[0] value to int: %v", err)
				}
			}
			// 找到比 queryRange[0] 大且差距最小的值
			tempIndex = binarySearchClosest(localClusterInt, queryRangeInt, true)
			//log.Printf("Left queryRangeInt: %d, tempIndex: %d, localClusterInt[x]: %d", queryRangeInt, tempIndex, localClusterInt[tempIndex])
		}
		tempToken := localCluster[0][tempIndex]
		sp.FlagEmpty = append(sp.FlagEmpty, tempToken) //若没有，则找到最接近的整数值作为查询关键字，然后生成token
		serverTokens = append(serverTokens, tempToken)
		sp.Flags = append(sp.Flags, "l") // 标记左边界需要查询\
	}

	if queryRange[1] != localCluster[len(localCluster)-1][len(localCluster[len(localCluster)-1])-1] { // 如果查询的“右边界”值不是某个区间的“右边界”值
		tempIndex := indexOf(localCluster[len(localCluster)-1], queryRange[1]) //区间有边界不需要左开区间处理，也就是不-1
		if tempIndex < 0 {                                                     //找到最接近的整数值作为查询关键字，然后生成token
			// 将 queryRange[1] 转换为 int
			queryRangeInt, err := strconv.Atoi(queryRange[1])
			if err != nil {
				log.Fatalf("Failed to convert queryRange[1] to int: %v", err)
			}

			// 将 localCluster[len(localCluster)-1] 转换为 []int
			localClusterInt := make([]int, len(localCluster[len(localCluster)-1]))
			for i, v := range localCluster[len(localCluster)-1] {
				localClusterInt[i], err = strconv.Atoi(v)
				if err != nil {
					log.Fatalf("Failed to convert localCluster[len(localCluster)-1] value to int: %v", err)
				}
			}

			// 找到比 queryRange[1] 小且差距最小的值
			tempIndex = binarySearchClosest(localClusterInt, queryRangeInt, false)
			//log.Printf("Right queryRangeInt: %d, tempIndex: %d, localClusterInt[x]: %d", queryRangeInt, tempIndex, localClusterInt[tempIndex])
		}
		tempToken := localCluster[len(localCluster)-1][tempIndex]
		sp.FlagEmpty = append(sp.FlagEmpty, tempToken) //若没有，则找到最接近的整数值作为查询关键字，然后生成token
		serverTokens = append(serverTokens, tempToken)
		sp.Flags = append(sp.Flags, "r") // 标记右边界需要查询
	}
	if p1 == p2 && len(sp.FlagEmpty) == 2 && sp.FlagEmpty[0] == sp.FlagEmpty[1] {
		//fmt.Println("Target query is in empty range!")
		return []string{}, nil
	}

	// 对 serverTokens 进行哈希处理
	hashedTokens := []string{}
	for _, token := range serverTokens {
		hashed := hex.EncodeToString(sp.H1([]byte(token)))
		hashedTokens = append(hashedTokens, hashed)
		//log.Printf("Generated token for %v: %v", token, hashed)
	}

	return hashedTokens, nil
}

func (sp *OurScheme) SearchTokens(tokens []string) [][]byte {
	searchResult := [][]byte{}
	for _, token := range tokens {
		// 从加密数据库中获取与 token 对应的加密位图
		if value, ok := sp.EDB[token]; ok {
			searchResult = append(searchResult, value)
			//log.Printf("Token found in EDB: %v, Encrypted result: %v", token, value)
		} else {
			log.Printf("Token not found in EDB: %v", token)
		}
	}
	return searchResult
}

func (sp *OurScheme) LocalSearch(searchResult [][]byte, tokens []string) ([]int, error) {
	clusterFlist := sp.ClusterFlist // 分区的文件列表
	finalResult := []int{}          // 搜索结果文件 ID 列表
	// 获取查询范围对应的分区位置
	p1, p2 := sp.LocalPosition[0], sp.LocalPosition[1]
	//log.Printf("Performing local search with LocalPosition: %v, Flags: %v", sp.LocalPosition, sp.Flags)

	// 如果没有服务器返回的加密结果，直接返回分区内的文件
	if len(searchResult) == 0 {
		//log.Printf("No encrypted results from server, returning all files in range.")
		for _, fileList := range clusterFlist[p1 : p2+1] {
			finalResult = append(finalResult, fileList...)
		}
		return finalResult, nil
	}

	// 生成字符串,1的个数为该分区包含的文档标识符个数，并转换为 byte[]
	fullOneBytes := []byte(strings.Repeat("1", len(sp.ClusterFlist[p1])) + strings.Repeat("0", sp.BsLength-len(sp.ClusterFlist[p1])))

	// 如果有两个加密结果（左边界和右边界）
	if len(searchResult) == 2 {
		decResult := [][]byte{
			xorBytesWithPadding(searchResult[0], sp.KeywordToSK[tokens[0]], sp.L), // 解密左边界
			xorBytesWithPadding(searchResult[1], sp.KeywordToSK[tokens[1]], sp.L), // 解密右边界
		}
		//log.Printf("Decrypted results: Left: %v, Right: %v", decResult[0], decResult[1])

		if p1 == p2 { // 单分区处理
			compBitmap := xorBytesWithPadding(decResult[0], decResult[1], sp.L) // 用异或计算，合并位图
			//log.Printf("Combined bitmap for single partition: %v", compBitmap)
			finalResult = append(finalResult, sp.parseFileID_for_01(compBitmap, clusterFlist[p1])...)
		} else { // 多分区处理,需要使用特殊parse解析01串
			leftBitmap := xorBytesWithPadding(decResult[0], fullOneBytes, sp.L)
			//log.Printf("Left bitmap: %v", leftBitmap)
			finalResult = append(finalResult, sp.parseFileID_for_01(leftBitmap, clusterFlist[p1])...)

			rightBitmap := decResult[1]
			//log.Printf("Right bitmap: %v", rightBitmap)
			finalResult = append(finalResult, sp.parseFileID(rightBitmap, clusterFlist[p2])...)

			// 处理中间分区的文件
			for _, fileList := range clusterFlist[p1+1 : p2] {
				finalResult = append(finalResult, fileList...)
			}
		}
	} else if len(searchResult) == 1 { // 单边界情况
		//log.Printf("sp.KeywordToSK[tokens[0]]: %v", sp.KeywordToSK[tokens[0]])
		decResult := xorBytesWithPadding(searchResult[0], sp.KeywordToSK[tokens[0]], sp.L)
		//log.Printf("Decrypted result for single token: %v", decResult)
		if contains(sp.Flags, "l") { // 处理左边界，需要使用特殊parse解析01串
			leftBitmap := xorBytesWithPadding(decResult, fullOneBytes, sp.L)
			//log.Printf("Left bitmap for single token: %v", leftBitmap)
			finalResult = append(finalResult, sp.parseFileID_for_01(leftBitmap, clusterFlist[p1])...)
			for _, fileList := range clusterFlist[p1+1 : p2+1] {
				finalResult = append(finalResult, fileList...)
			}
		}
		if contains(sp.Flags, "r") { // 处理右边界
			rightBitmap := decResult
			//log.Printf("Right bitmap for single token: %v", rightBitmap)
			finalResult = append(finalResult, sp.parseFileID(rightBitmap, clusterFlist[p2])...)
			for _, fileList := range clusterFlist[p1:p2] {
				finalResult = append(finalResult, fileList...)
			}
		}
	}

	return finalResult, nil
}

// searchTree 在本地树中查找关键词的位置
func (sp *OurScheme) searchTree(queryValue string) (int, error) {
	// 将查询值转化为整数（假设查询值是字符串格式的数字）
	queryValueInt, err := strconv.ParseInt(queryValue, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("无法将查询值转换为整数: %v", err)
	}

	node := "0" // 初始化为树的根节点
	// 获取当前节点的值
	nodeValue, ok := sp.LocalTree[node]
	if !ok {
		return 0, fmt.Errorf("无法找到节点 %s", node)
	}
	if queryValueInt > nodeValue[1] || queryValueInt < nodeValue[0] {
		return 0, fmt.Errorf("queryValueInt %d 在范围[%d,%d]之外，无法找到节点", queryValueInt, nodeValue[0], nodeValue[1])
	}

	height := int(math.Ceil(math.Log2(float64(len(sp.ClusterFlist)))))
	for i := 0; i < height; i++ {
		valuelr := sp.LocalTree[node+"0"][1]
		valuerl := sp.LocalTree[node+"1"][0]
		// 比较查询值与当前节点的值
		if queryValueInt > valuelr {
			node += "1" // 如果大于当前节点，向右子树移动
		} else if queryValueInt < valuerl {
			node += "0" // 如果小于或等于当前节点，向左子树移动
		}
	}
	valuelf := sp.LocalTree[node]
	valuelf[0] += 1
	// 将二进制节点位置转换为整数
	position, err := strconv.ParseInt(node, 2, 64)
	if err != nil {
		return 0, fmt.Errorf("解析查询值出错：%v", err)
	}

	return int(position), nil
}

// 辅助函数：生成位图
func (sp *OurScheme) generateBitmap(group []int) []byte {
	// 计算 sp.L - len(group) 的值，并确保它不为负数
	remainingLength := sp.L - len(group)
	if remainingLength < 0 {
		remainingLength = 0
		// 打印错误信息
		//log.Printf("错误：group 的长度大于 sp.L，remainingLength 被置为 0")
	}

	// 生成位图字符串
	bitString := strings.Repeat("1", len(group)) + strings.Repeat("0", remainingLength)

	// 返回生成的位图字节切片
	return []byte(bitString)
}

// 辅助函数：生成位图
func (sp *OurScheme) generateBitmap1(group []int) []byte {
	bitString := strings.Repeat("1", len(group)) + strings.Repeat("0", sp.L-len(group))
	//log.Printf("bitString: %s", bitString)
	//bitmap, _ := strconv.ParseUint(bitString, 2, 64)
	//log.Printf("bitmap_int: %d", bitmap)
	return []byte(bitString)
}
func xorBytesWithPadding(a, b []byte, bytelen int) []byte {
	// 检查 bytelen 是否合法
	if bytelen <= 0 {
		return []byte{}
	}
	// 如果 a 和 b 完全相同，则直接返回 a
	if len(a) == len(b) && bytes.Equal(a, b) {
		return a
	}
	// 截取 a 和 b 的最低 bytelen 个字节（如果长度不足，填充 0）
	truncatedA := make([]byte, bytelen)
	truncatedB := make([]byte, bytelen)

	// 从后往前拷贝数据
	copy(truncatedA[max(0, bytelen-len(a)):], a[max(0, len(a)-bytelen):])
	copy(truncatedB[max(0, bytelen-len(b)):], b[max(0, len(b)-bytelen):])

	// 执行异或操作
	result := make([]byte, bytelen)
	for i := 0; i < bytelen; i++ {
		result[i] = truncatedA[i] ^ truncatedB[i]
	}

	return result
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func (sp *OurScheme) parseFileID_for_01(bitmap []uint8, dbList []int) []int {
	result := []int{}
	dblen := len(dbList)
	// 遍历 bitmap 的每一位
	for byteIndex, b := range bitmap {
		if b == 1 && byteIndex < dblen { // 检查当前位是否为 1
			result = append(result, dbList[byteIndex])
		}
	}

	return result
}
func (sp *OurScheme) parseFileID(bitmap []byte, dbList []int) []int {
	result := []int{}

	// 将位图直接转换为字符形式的字符串
	bitString := ""
	for _, b := range bitmap {
		// 检查 b 是否是 ASCII 可显示字符，如果是则直接转换为字符，否则转换为二进制字符串
		if b >= 32 && b <= 126 { // ASCII 可见字符范围
			bitString += string(b)
		} else {
			bitString += fmt.Sprintf("%08b", b) // 转换为 8 位二进制字符串
		}
	}

	// 打印 bitString
	//fmt.Printf("BitString (converted): %s\n", bitString)

	// 只保留从右往左数的 L 位
	if len(bitString) > sp.L {
		bitString = bitString[len(bitString)-sp.L:] // 截取最后 L 位
	}

	// 遍历保留的 L 位，提取对应位置的文件 ID
	for i, bit := range bitString {
		if i >= len(dbList) { // 超出文件列表长度则停止
			break
		}
		if bit == '1' { // 如果位为 '1'，则添加对应的文件 ID
			result = append(result, dbList[i])
		}
	}

	return result
}
func indexOf(slice []string, value string) int {
	for i, v := range slice {
		if v == value {
			return i
		}
	}
	return -1 // 未找到返回 -1
}
func contains(slice []string, item string) bool {
	for _, v := range slice {
		if v == item {
			return true
		}
	}
	return false
}
func listSearchClosest(slice []int, value int, findLarger bool) int {
	if len(slice) == 0 {
		return -1 // 如果 slice 为空，返回 -1 表示无效
	}

	closest := -1

	if findLarger {
		// 向右逐个遍历，查找比 value 大的最近值
		for i, v := range slice {
			if v >= value { // 找到第一个大于或等于 value 的值
				if v == value {
					return i // 找到相等值，直接返回下标
				}
				if closest == -1 || v < slice[closest] {
					closest = i
				}
			}
		}
	} else {
		// 向左逐个遍历，查找比 value 小的最近值
		for i := len(slice) - 1; i >= 0; i-- {
			v := slice[i]
			if v <= value { // 找到第一个小于或等于 value 的值
				if v == value {
					return i // 找到相等值，直接返回下标
				}
				if closest == -1 || v > slice[closest] {
					closest = i
				}
			}
		}
	}

	return closest
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
