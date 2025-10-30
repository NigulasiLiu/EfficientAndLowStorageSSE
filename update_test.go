package main

import (
	"EfficientAndLowStroageSSE/FB_RSSE"
	"EfficientAndLowStroageSSE/VH_RSSE/OurScheme"
	"math/big"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"
)

// 测试配置
const (
	L            = 6424                               // 固定L值
	resultFile   = "update_time_results.txt"          // 结果输出文件
	datasetDir   = "dataset/"                         // 数据集目录
	fileTemplate = "Gowalla_invertedIndex_new_%d.txt" // 数据集文件名模板
)

// 待测试的关键字数量（仅小数据集）
var testMs = []int{5000, 10000, 15000, 20000}

// 提取并排序关键字
func getSortedKeywords(invertedIndex map[string][]int) []string {
	keywords := make([]string, 0, len(invertedIndex))
	for k := range invertedIndex {
		keywords = append(keywords, k)
	}
	// 按数值排序（与项目中逻辑一致）
	sort.Slice(keywords, func(i, j int) bool {
		ki, _ := strconv.ParseInt(keywords[i], 10, 64)
		kj, _ := strconv.ParseInt(keywords[j], 10, 64)
		return ki < kj
	})
	return keywords
}

// 生成测试用的更新位图
func generateUpdateBitmap(docIDs []int) *big.Int {
	bs := big.NewInt(0)
	for _, docID := range docIDs {
		bs.SetBit(bs, docID, 1)
	}
	return bs
}

// 写入测试结果到文件
func writeResult(content string) error {
	f, err := os.OpenFile(resultFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(content + "\n")
	return err
}

func TestUpdatePerformance(t *testing.T) {
	// 设置参数
	files := []string{
		"dataset/Gowalla_invertedIndex_new_25000.txt",
		"dataset/Gowalla_invertedIndex_new_20000.txt",
		"dataset/Gowalla_invertedIndex_new_15000.txt",
		"dataset/Gowalla_invertedIndex_new_10000.txt",
		"dataset/Gowalla_invertedIndex_new_5000.txt",
	}
	indexNum := []int{25000, 20000, 15000, 10000, 5000}
	FB_BsLen := 15
	LValues := []int{6424}
	resultsDir := "results"

	rand.Seed(time.Now().UnixNano())

	for fileIndex, file := range files {
		// 加载索引和关键词（无日志）
		invertedIndex, _ := loadInvertedIndex(file)
		keywords := extractKeywordsFromIndex(invertedIndex)
		sortedKeywords := sortKeywords(invertedIndex)

		for _, L := range LValues {
			// 初始化方案（无日志）
			ours := OurScheme.Setup(L)
			fbDsseParams := FB_RSSE.Setup(FB_BsLen)

			// 索引构建（仅保留耗时日志）
			startTime := time.Now()
			_ = ours.BuildIndex(invertedIndex, sortedKeywords)
			buildOursDur := time.Since(startTime).Nanoseconds()

			startTime = time.Now()
			_ = fbDsseParams.BuildIndexMock(invertedIndex, sortedKeywords)
			buildFBDur := time.Since(startTime).Nanoseconds()

			// --------------------------
			// 更新测试阶段（核心日志保留）
			// --------------------------
			const updateTotalTimes = 10
			t.Logf("关键字数量=%d, L=%d, 更新轮次=%d", indexNum[fileIndex], L, updateTotalTimes)

			// OurScheme 更新测试
			var totalOursDur int64 = 0
			for updateRound := 0; updateRound < updateTotalTimes; updateRound++ {
				kw := strconv.Itoa(keywords[rand.Intn(len(keywords))])
				start := time.Now()
				_ = ours.Update(kw, generateRandomDocIDsBigInt(10))
				totalOursDur += time.Since(start).Nanoseconds()
			}
			avgOursDur := totalOursDur / updateTotalTimes
			t.Logf("OurScheme 平均更新耗时=%d ns", avgOursDur)

			// FB_DSSE 更新测试
			t.Logf("开始 FB_DSSE 更新测试...")
			var totalFBDur int64 = 0
			for updateRound := 0; updateRound < updateTotalTimes; updateRound++ {
				kw := strconv.Itoa(keywords[rand.Intn(len(keywords))])
				docIDs := generateRandomDocIDsBigInt(10) // 生成长度为10的数组

				if len(docIDs) == 0 {
					t.Errorf("FB_DSSE 第 %d 次更新: 文档ID为空", updateRound+1)
					continue
				}

				start := time.Now()

				// 遍历数组中的每个元素，分别调用 UpdateBigInt
				for _, docID := range docIDs {
					if err := fbDsseParams.UpdateBigInt(kw, docID); err != nil {
						t.Errorf("FB_DSSE 第 %d 次更新失败 (docID=%v): %v", updateRound+1, docID, err)
						continue
					}
				}

				totalFBDur += time.Since(start).Nanoseconds()
			}
			avgFBDur := totalFBDur / updateTotalTimes
			t.Logf("FB_DSSE 平均更新耗时=%d ns", avgFBDur)

			// 保存结果（无日志）
			saveResult(resultsDir, indexNum[fileIndex], L, map[string]interface{}{
				"keyword_count":   indexNum[fileIndex],
				"L":               L,
				"build_ours":      buildOursDur,
				"build_fb":        buildFBDur,
				"update_ours_avg": avgOursDur,
				"update_fb_avg":   avgFBDur,
			})
		}
	}
}

// 辅助函数：确保目录存在（如果不存在则创建）
func ensureDirExists(dir string) error {
	_, err := os.Stat(dir)
	if os.IsNotExist(err) {
		return os.MkdirAll(dir, 0755)
	}
	return err
}

// 辅助函数：保存测试结果（根据实际需求实现）
func saveResult(dir string, count, L int, data map[string]interface{}) error {
	// 示例：可将结果保存为JSON文件
	// 实际实现需引入encoding/json和os包
	return nil
}

// generateRandomDocIDs 生成指定数量的随机文档ID（int类型），仅依赖math/rand
func generateRandomDocIDs(count int) []int {
	docIDs := make([]int, count)
	// 初始化随机种子（全局只需初始化一次，若外部已初始化可移除）
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < count; i++ {
		// 生成1~1e3范围的随机int（确保非零）
		randInt := rand.Intn(1e3) // 0~999
		if randInt == 0 {
			randInt = 1 // 避免0值，确保ID有效
		}
		docIDs[i] = randInt
	}
	return docIDs
}

// generateRandomDocIDs 生成指定数量的随机文档ID（big.Int类型），仅依赖math/rand
func generateRandomDocIDsBigInt(count int) []*big.Int {
	docIDs := make([]*big.Int, count)
	// 初始化math/rand随机种子（确保每次运行生成不同随机序列）
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < count; i++ {
		// 步骤1：用math/rand生成int64范围的随机数（0 ~ 1e9-1）
		//randInt64 := int64(rand.Intn(1e9))
		randInt64 := int64(rand.Intn(1e9))
		// 步骤2：将int64转为big.Int类型
		randBigInt := big.NewInt(randInt64)
		// 步骤3：确保文档ID非零（若随机数为0则加1，否则保持原数）
		if randBigInt.Sign() == 0 {
			randBigInt.Add(randBigInt, big.NewInt(1))
		}
		docIDs[i] = randBigInt
	}
	return docIDs
}
