# 基于代码进化的结构化数据自动化挖掘系统

针对任意数值、文本模态结构化数据，实现计算逻辑自动化演进的智能体框架。

通过 **“代码进化 - 增强 - 评估 - 反馈”** 闭环机制，系统能够自主完成特征挖掘、特征增强，并显著提升主模型的指标增益（如 AUC、KS）。

---

## 🌟 核心理念

本项目摒弃了传统的固定特征工程算子，采用动态进化的思路：
1.  **代码进化**：利用 LLM 根据当前数据特征生成复杂的复合计算算子。
2.  **增强**：局部代码增强（如算子系数），增强代码处理能力。
3.  **评估**：在验证集上实时评估生成代码的效果指标。
4.  **反馈**：将评估结果反馈给 LLM，指导下一代代码的进化。

---

## 🏗️ 系统架构

### 1. 系统总体结构
系统由进化引擎、增强环节、评估环境以及反馈环路组成。
![系统结构图](./系统结构.png)

### 2. 外接模型库服务
支持在生成的进化代码中动态调用外部深度学习服务（如 NLP 语义分析、向量检索等）。
![算子库调用图](./算子库调用.png)

---

## 🚀 技术创新点

相较于**AlphaEvolve** 等方案，本项目的技术创新点如下：

> **对比分析图：**
> ![对比图](./对比.png)

---

## 🔍 挖掘模式

支持对任意数值、文本模态的结构化数据进行深度挖掘。
![挖掘模式图](./模式.png)

---

## 💻 生成代码示例 (Case Study)

### 案例 1：反备注信息深度挖掘

```python
    def calculate_probability(data):
        import math
        import numpy as np
        import pandas as pd
        import re
        base_score = 0.0

        # 处理B字段：applist，按'^'分割计数
        B = data.get('B')
        if isinstance(B, str):
            b_count = len([x for x in B.split('^') if x.strip()])
        else:
            b_count = 0
        log_b = math.log(b_count + np.float64(0.1))  # 避免log(0)

        # 处理D字段：人行单位，按'^'分割计数
        D = data.get('D')
        if isinstance(D, str):
            d_count = len([x for x in D.split('^') if x.strip()])
        else:
            d_count = 0
        log_d = math.log(d_count + np.float64(0.1))  # 避免log(0)

        # 处理F字段：命中职业，按';'分割计数
        F = data.get('F')
        if isinstance(F, str):
            f_count = len([x for x in F.split(';') if x.strip()])
        else:
            f_count = 0

        # 处理J字段：命中关键词，按'^'分割计数
        J = data.get('J')
        if isinstance(J, str):
            j_count = len([x for x in J.split('^') if x.strip()])
        else:
            j_count = 0

        # 处理G字段：命中规则id，浮点值或0
        G = data.get('G')
        try:
            g_val = float(G) if not pd.isna(G) else 0.0
        except (ValueError, TypeError):
            g_val = 0.0

        # 处理I字段：命中规则频次，浮点值或0
        I = data.get('I')
        try:
            i_val = float(I) if not pd.isna(I) else 0.0
        except (ValueError, TypeError):
            i_val = 0.0

        # 处理H字段：命中时间，浮点值或0
        H = data.get('H')
        try:
            h_val = float(H) if not pd.isna(H) else 0.0
        except (ValueError, TypeError):
            h_val = 0.0

        # 计算正负特征及交互值
        negative_feature = log_b * np.float64(3.235206813194917) + log_d * np.float64(10.0)  # 负特征：B和D的对数和
        positive_feature = f_count * np.float64(11.72937046478163) + j_count * np.float64(7.75378475770262) + g_val * np.float64(10.0) + i_val * np.float64(0.1) + h_val * np.float64(10.0)  # 正特征：加权和
        interaction = positive_feature * np.float64(-8.197161404989746) + negative_feature * np.float64(-10.0)  # 交互值

        # 平方根复合算子计算调整分
        score_adjust = np.float64(0.1) * math.sqrt(abs(interaction)) * np.sign(interaction)

        # 新增：基于D字段正关键词的调整分（仅在交互值为负时生效）
        d_positive_keywords = ["机电", "公司", "集团", "银行", "发电", "健康", "管理", "实业", "家具", "幼儿园"]
        d_positive_count = np.float64(0.6273060677820665)
        if isinstance(D, str):
            d_parts = [part.strip() for part in D.split('^') if part.strip()]
            for part in d_parts:
                if any(kw in part for kw in d_positive_keywords):
                    d_positive_count += np.float64(0.1)
        if interaction < 0:
            score_adjust += np.float64(0.7498485170716492) * math.log(d_count + np.float64(0.01)) * d_positive_count

        # 新增：多字段复杂交互增强调整
        # 1. 处理G/I/H多值字段：统计';'分割的有效条目数
        new_g_count = 0
        if isinstance(G, str):
            new_g_count = len([x for x in G.split(';') if x.strip()])
        new_i_count = 0
        if isinstance(I, str):
            new_i_count = len([x for x in I.split(';') if x.strip()])
        new_h_count = 0
        if isinstance(H, str):
            new_h_count = len([x for x in H.split(';') if x.strip()])

        # 2. 处理A字段：统计含正类关键词的条目数（最多max_a_positive_count个）
        a_positive_count = 0
        a_positive_keywords = ["经理", "老板", "电工", "装修", "公司", "人事部", "师傅", "总", "厂家", "安装", "装饰", "工程", "施工", "姑爷"]
        a_entries = []
        if isinstance(data.get('A'), str):
            a_entries = [entry.strip() for entry in data.get('A').split('^^^') if entry.strip()]
            for entry in a_entries:
                if any(kw in entry for kw in a_positive_keywords):
                    a_positive_count += 1
            a_positive_count = min(a_positive_count, int(np.float64(1.0)))  # 限制最大计数避免过拟合

        # 3. 复杂交互与复合算子生成增强调整分
        interaction_enhanced = (new_g_count + new_i_count + new_h_count) * math.log(a_positive_count + np.float64(0.1)) + positive_feature * np.float64(0.01)
        adjustment_enhanced = np.float64(0.6280177531805928) * math.sqrt(abs(interaction_enhanced)) * np.sign(interaction_enhanced)
        score_adjust += adjustment_enhanced

        # 新增：基于A字段条目数与D字段正关键词计数的复杂交互惩罚项（提升正负样本区分度）
        a_entry_count = len(a_entries) if isinstance(data.get('A'), str) else np.float64(0.04562412600355873)
        interaction_adjust = a_entry_count * d_positive_count
        penalty_adjust = np.float64(0.4819496409372016) * math.sqrt(abs(interaction_adjust))  # 复合算子：平方根
        score_adjust += np.float64(-0.7660486194226621) * penalty_adjust

        # 新增：负向职业与不稳定单位结合规则命中强度的复合惩罚项（参数化部分）
        # 1. 定义负向职业关键词及不稳定单位关键词
        negative_occupation_keywords = ["货车司机", "网约车司机", "装修工人", "修车工", "维修工人"]
        unstable_d_keywords = ["自由职业", "无单位", "未知"]
        # 2. 统计F字段负向职业计数（参数化sum系数）
        f_negative_count = 0
        if isinstance(F, str):
            f_parts = [part.strip() for part in F.split(';') if part.strip()]
            f_negative_count = sum(np.float64(1.854187738333358) for part in f_parts if any(kw in part for kw in negative_occupation_keywords))
        # 3. 判断D字段是否为不稳定单位
        is_unstable_d = False
        if isinstance(D, str):
            d_str = D.lower()  # 大小写不敏感匹配
            is_unstable_d = any(kw in d_str for kw in unstable_d_keywords)
        # 4. 计算规则命中总次数
        total_rule_hits = new_g_count + new_i_count + new_h_count
        # 5. 计算复合惩罚项（仅当同时满足负向职业和不稳定单位时生效，参数化阈值、log_epsilon、系数、符号）
        if f_negative_count > np.float64(0.9745840075695658) and is_unstable_d:
            interaction_penalty = total_rule_hits * math.log(f_negative_count + np.float64(0.46656575748808493))
            penalty_term = np.float64(0.9956168709414356) * math.sqrt(abs(interaction_penalty)) * np.sign(interaction_penalty)
            score_adjust += np.float64(-1.9915807728785822) * penalty_term  # 惩罚项降低分数（参数化符号）

        # ------------------------------ 新增复合交互调整项 ------------------------------
        # 1. D字段稳定性评分
        stable_d_keywords = ["公司", "集团", "幼儿园", "税务局", "装饰设计室", "科技有限公司", "贸易有限公司", "建筑材料有限公司"]
        unstable_d_keywords_ext = unstable_d_keywords + ["暂未提供", "弹性工作者", "无单位"]
        d_stability_score = np.float64(0.5)
        if isinstance(D, str):
            d_lower = D.lower()
            if any(kw.lower() in d_lower for kw in stable_d_keywords):
                d_stability_score = np.float64(0.5)
            elif any(kw.lower() in d_lower for kw in unstable_d_keywords_ext):
                d_stability_score = np.float64(-1.012935344935292)
        # 2. A字段正头衔计数（扩展限制，避免过拟合）
        a_positive_count_ext = min(a_positive_count, int(np.float64(3.464330223167567)))  # 限制最大计数避免过拟合
        # 3. F字段负向职业计数（去重统计）
        f_negative_unique = set()
        if isinstance(F, str):
            for part in f_parts:
                if any(kw in part for kw in negative_occupation_keywords):
                    f_negative_unique.add(part.strip())
        f_negative_unique_count = len(f_negative_unique)
        # 4. 复合调整项计算
        stability_reward = 0.0
        if d_stability_score > 0:
            stability_reward = np.float64(0.1) * math.log(np.float64(5.0) + a_positive_count_ext) * (d_stability_score ** np.float64(3.0))
        instability_penalty = 0.0
        if d_stability_score < 0:
            instability_penalty = np.float64(-2.0) * math.sqrt(f_negative_unique_count + np.float64(5.0)) * math.log(np.float64(0.1) + abs(d_stability_score))
        # 5. 加入分数调整
        score_adjust += stability_reward + instability_penalty
        # --------------------------------------------------------------------------------

        # 新增：负向职业与单位稳定性的复合缓解/惩罚调整（增强正负样本区分度）
        f_negative_count_raw = np.float64(-0.5)
        if isinstance(F, str):
            f_parts = [part.strip() for part in F.split(';') if part.strip()]
            f_negative_count_raw = sum(np.float64(5.0) for part in f_parts if any(kw in part for kw in negative_occupation_keywords))
        if f_negative_count_raw > 0:
            # 复合算子：平方根控制增长 + 稳定性得分引导方向 + 系数控制强度
            stability_interaction = np.float64(1.1424054899131297) * math.sqrt(f_negative_count_raw + np.float64(10.0)) * d_stability_score
            score_adjust += stability_interaction

        # ------------------------------ 新增：负向应用-不稳定单位-规则命中复合惩罚项 ------------------------------
        # 1. 统计B字段贷款类负面应用数量
        negative_app_keywords = ["360借条", "安逸花", "有钱花", "奇富借条", "国美易卡", "信用飞"]  # 扩展负面应用关键词
        b_negative_app_count = 0
        if isinstance(B, str):
            b_apps = [app.strip().lower() for app in B.split('^') if app.strip()]
            b_negative_app_count = sum(1 for app in b_apps if any(kw.lower() in app for kw in negative_app_keywords))
        # 2. 计算不稳定单位得分（-1=不稳定，0=稳定，NaN=-0.5）
        unstable_d_score = np.float64(0.9241456886204249)
        if isinstance(D, str):
            d_lower = D.lower()
            if any(kw in d_lower for kw in unstable_d_keywords_ext):
                unstable_d_score = np.float64(-4.5354958728000225)
            else:
                unstable_d_score = np.float64(1.0754485190143832)
        else:
            unstable_d_score = np.float64(-3.294758763127085)
        # 3. 计算F字段负向职业计数（去重+加权）
        f_negative_weighted = np.float64(0.0)
        if isinstance(F, str):
            f_unique_parts = set(part.strip() for part in F.split(';') if part.strip())
            f_negative_weighted = sum(np.float64(0.41875280562786965) for part in f_unique_parts if any(kw in part for kw in negative_occupation_keywords))
        # 4. 复合惩罚项计算：对数控制权重+乘积交互+平方根抑制极端值
        risk_interaction = b_negative_app_count * f_negative_weighted * unstable_d_score
        if risk_interaction < 0:  # 仅当存在风险信号组合时生效
            log_risk = math.log(abs(risk_interaction) + np.float64(9.4893665171608))  # 避免log(0)
            composite_penalty = np.float64(-0.44024287256186234) * math.sqrt(abs(log_risk)) * np.sign(risk_interaction)
            score_adjust += composite_penalty
        # --------------------------------------------------------------------------------

        # ------------------------------ 新增：负向职业+不稳定单位的正信号救援调整 ------------------------------
        # 1. 增强A字段正信号：关键词加权+去重计数（避免过拟合）
        high_a_weight = np.float64(4.039033239601014)
        medium_a_weight = np.float64(2.748830422249555)
        low_a_weight = np.float64(1.7034272246099866)
        a_positive_max = np.float64(1.9254153909118856)

        a_positive_weights = {
            "经理": high_a_weight, "老板": high_a_weight, "总": high_a_weight, "人事部": high_a_weight,  # 高权重
            "电工": medium_a_weight, "装修": medium_a_weight, "公司": medium_a_weight, "厂家": medium_a_weight,  # 中权重
            "装饰": medium_a_weight, "工程": medium_a_weight, "施工": medium_a_weight,
            "师傅": low_a_weight, "安装": low_a_weight, "姑爷": low_a_weight  # 低权重
        }

        enhanced_a_positive = 0.0
        if isinstance(data.get('A'), str):
            for entry in a_entries:
                for kw in a_positive_weights:
                    if kw in entry:
                        enhanced_a_positive += a_positive_weights[kw]
            enhanced_a_positive = min(enhanced_a_positive, a_positive_max)  # 上限控制

        # 2. F字段负向职业唯一严重性评分
        high_occ_severity = np.float64(1.1519228096937388)
        medium_occ_severity = np.float64(2.3040478107715763)

        negative_occupation_severity = {
            "货车司机": high_occ_severity, "网约车司机": high_occ_severity,  # 高严重性
            "装修工人": medium_occ_severity, "修车工": medium_occ_severity, "维修工人": medium_occ_severity  # 中严重性
        }

        f_negative_severity = 0.0
        if isinstance(F, str):
            f_unique_negative = set()
            for part in f_parts:
                for kw in negative_occupation_severity:
                    if kw in part:
                        f_unique_negative.add(kw)
            f_negative_severity = sum(negative_occupation_severity[kw] for kw in f_unique_negative)

        # 3. D字段不稳定严重性评分
        d_high_instability = np.float64(2.153660552702697)
        d_medium_instability = np.float64(3.0)
        d_low_instability = np.float64(-0.1)
        d_nan_instability = np.float64(1.1719043946317325)

        d_instability_severity = 0.0
        if isinstance(D, str):
            d_lower = D.lower()
            if any(kw in d_lower for kw in ["无单位", "自由职业"]):
                d_instability_severity = d_high_instability
            elif any(kw in d_lower for kw in ["未知", "暂未提供"]):
                d_instability_severity = d_medium_instability
            else:
                d_instability_severity = d_low_instability
        else:
            d_instability_severity = d_nan_instability  # NaN视为中度不稳定

        # 4. 复合救援分计算（仅当现有交互为负时生效）
        log_epsilon = np.float64(0.41417380361264966)
        rescue_coeff = np.float64(1.8327161485404033)

        rescue_score = 0.0
        if interaction < 0:  # 现有模型正在 penalizing
            if f_negative_severity > 0 and d_instability_severity > 0:
                # 对数增强信号 + 平方根控制增长 + 系数调整强度
                log_negative = math.log(log_epsilon + f_negative_severity)  # 避免log(0)
                log_d_instability = math.log(log_epsilon + d_instability_severity)
                combined_signal = enhanced_a_positive * log_negative * log_d_instability
                if combined_signal > 0:
                    rescue_score = rescue_coeff * math.sqrt(combined_signal)  # 系数经样本验证
        score_adjust += rescue_score
        # --------------------------------------------------------------------------------

        # ------------------------------ 新增：稳定单位与风险信号复合惩罚项（核心优化）------------------------------
        # 1. 定义D字段稳定性关键词（高/中/低/不稳定）
        high_stable_d_keywords = ["公司", "集团", "有限公司", "学院", "物业", "酒店", "医院", "学校", "政府", "供电所", "保利", "万科"]
        medium_stable_d_keywords = ["村委会", "村民委员会", "工厂", "商行", "超市"]
        # 2. 计算D字段稳定性得分（数值化）
        d_stability_numeric = 0.0
        if isinstance(D, str):
            d_lower = D.lower()
            if any(kw in d_lower for kw in high_stable_d_keywords):
                d_stability_numeric = np.float64(4.908684351454522)  # 高稳定性：参数化
            elif any(kw in d_lower for kw in medium_stable_d_keywords):
                d_stability_numeric = np.float64(-0.9134527497949936)  # 中等稳定性：参数化
            elif any(kw in d_lower for kw in unstable_d_keywords_ext):
                d_stability_numeric = np.float64(2.09646882910958)  # 低稳定性（不稳定）：参数化
            else:
                d_stability_numeric = np.float64(-1.9177725820937517)  # 中性：参数化
        # 3. 计算B字段负面应用风险得分（加权计数）
        negative_apps = ["360借条", "安逸花", "有钱花", "奇富借条", "国美易卡", "信用飞", "分期乐", "拍拍贷借款", "你我贷借款", "宜享花", "畅行花"]
        b_negative_risk = 0.0
        if isinstance(B, str):
            b_apps = [app.strip().lower() for app in B.split('^') if app.strip()]
            b_negative_risk = sum(np.float64(0.0661928446545443) for app in b_apps if any(kw in app for kw in negative_apps))  # 加权降低单一应用影响：参数化权重
        # 4. 计算F字段负向职业风险得分（唯一计数）
        f_negative_risk = len(f_negative_unique) if isinstance(F, str) else 0.0  # 去重计数避免重复惩罚
        # 5. 总风险得分
        total_risk = b_negative_risk + f_negative_risk
        # 6. 计算复合惩罚项（仅针对高稳定性+高风险样本）
        stability_risk_penalty = 0.0
        if d_stability_numeric >= np.float64(0.9260665327145966) and total_risk >= np.float64(1.031190892762655):  # 阈值参数化
            epsilon = np.float64(0.21474267344107986)  # log epsilon参数化
            interaction_strength = d_stability_numeric * total_risk
            log_term = math.log(epsilon + interaction_strength)
            sqrt_term = math.sqrt(abs(log_term))
            stability_risk_penalty = np.float64(-2.763477503504929) * sqrt_term  # 惩罚系数参数化（包含符号）
        # 7. 加入分数调整
        score_adjust += stability_risk_penalty
        # --------------------------------------------------------------------------------

        # ------------------------------ 新增：负向职业-不稳定单位-正信号补偿复合调整项（增强正负区分度）------------------------------
        # 1. 计算负向职业风险得分（唯一严重性总和）
        f_negative_score = f_negative_severity  # 已存在：唯一负向职业的严重性总和
        # 2. 计算单位不稳定得分（数值越大越不稳定）
        d_instability_score = max(np.float64(0.2463169516499296), -d_stability_numeric)  # 稳定单位得分为0，不稳定单位得分>0（参数化下限）
        # 3. 计算正信号补偿得分（A增强正信号 + J关键词计数加权）
        positive_mitigation = enhanced_a_positive + (j_count * np.float64(0.1))  # J权重参数化平衡A的贡献
        # 4. 计算组合风险与补偿效果
        combined_risk = f_negative_score * d_instability_score
        mitigation_effect = positive_mitigation / (combined_risk + np.float64(7.3519044014974125e-06))  # 避免除零，epsilon参数化
        risk_net = combined_risk - mitigation_effect
        # 5. 复合惩罚项：仅当净风险为正时生效，平方根控制增长
        if risk_net > np.float64(-0.1):
            composite_penalty = np.float64(-1.0) * math.sqrt(risk_net) * np.sign(risk_net)  # 惩罚系数参数化
        else:
            composite_penalty = np.float64(-0.1)  # else分支惩罚参数化
        # 6. 加入分数调整
        score_adjust += composite_penalty
        # --------------------------------------------------------------------------------

        # 总分计算
        score_before_normalized = base_score + score_adjust

        prob = 1 / (1 + math.exp(-score_before_normalized))
        prob = max(0., min(1., prob))

        return score_before_normalized, prob

### 案例 2：贷后通话文本挖掘

```python
def calculate_probability(data):

        import math # 这一行不要改，且必须在calculate_probability函数中
        import numpy as np # 这一行不要改，必须在calculate_probability函数中
        import pandas as pd # 这一行不要改，必须在calculate_probability函数中
        import re # 这一行不要改，必须在calculate_probability函数中
        base_score = 0.0

        A = data.get('A')
        if isinstance(A, str) or A is None:
            text = A or ''
        else:
            try:
                text = str(float(A)) if not pd.isna(A) else ''
            except (ValueError, TypeError):
                text = ''
        if pd.isna(A):
            text = ''

        text_length = len(text)
        sentences = re.split(r'[。！？]', text)
        sentence_count = len([s for s in sentences if s.strip()])
        words = re.findall(r'[一-龥a-zA-Z0-9]+', text)

        repayment_words = {"还款", "还", "处理", "APP", "减免", "尽快", "可以", "行", "好的", "马上", "撤销", "已经", "会", "能", "找到", "登录", "操作", "结清", "搞定"}
        delay_words = {"无法接通", "留言", "没钱", "不是", "不知道", "过", "啊", "哦", "嗯", "拖延", "等一下", "明天", "后天", "以后", "不行", "不用", "别", "挂断", "没", "不"}
        overdue_words = {"逾期", "欠款", "催收", "拖欠", "未还", "账单", "贷款", "借条", "平台", "征信", "起诉", "法务", "黑名单", "违约", "催缴"}

        # 词频计算权重参数化
        word_weight = np.float64(0.1)
        R = sum(word_weight for word in words if word in repayment_words)
        D = sum(word_weight for word in delay_words if word in delay_words)
        O = sum(word_weight for word in overdue_words if word in overdue_words)

        # numerator计算参数化
        rd_coef = np.float64(10.0)
        log_text_offset = np.float64(0.1)
        numerator = rd_coef * (R - D) * math.log(text_length + log_text_offset) if text_length > 0 else 0.0

        # denominator计算参数化
        O_coef = np.float64(0.1)
        sqrt_sent_offset = np.float64(10.0)
        denom_min = np.float64(5.0)
        denominator_candidate = O_coef * O + math.sqrt(sentence_count + sqrt_sent_offset)
        # freeze 
        denominator = denominator_candidate if denominator_candidate != 0 else denom_min
        interaction = numerator / denominator # freeze 
        # 树状增长
        log_adj_offset = np.float64(10.0)
        exp_coef = np.float64(0.1)
        if interaction >= 0:
            adjustment = math.log(interaction + log_adj_offset)
        else:
            adjustment = exp_coef * math.exp(interaction)

        # adjustment权重参数化
        adj_weight = np.float64(0.1)
        score_before_normalized = base_score + adj_weight * adjustment

        # 新增：风险意愿平衡特征计算
        log_text_len_offset = np.float64(6.3376747124182815)
        log_text_len = math.log(text_length + log_text_len_offset) if text_length > 0 else 0.0  # 避免log(0)，平滑文本长度影响
        rwb = O * (R - D) * log_text_len  # 综合逾期感知、意愿差、文本长度的交互特征
        # 新增：复合函数调整风险意愿平衡特征
        new_adj_weight = np.float64(0.6031240412784725)
        log_rwb_offset = np.float64(0.13629295292296895)
        log_adj_multiplier = np.float64(9.802380210487746)
        exp_denom = np.float64(1.2865929728087535)
        exp_adj_multiplier = np.float64(0.8709141564883808)
        if rwb > 0:
            adjustment_new = math.log(rwb + log_rwb_offset) * log_adj_multiplier  # 正向信号用对数放大，增强正类区分度
        else:
            adjustment_new = math.exp(rwb / exp_denom) * exp_adj_multiplier  # 负向信号用指数衰减抑制，弱化负类干扰
        # 整合新调整项到原始分数
        score_before_normalized += new_adj_weight * adjustment_new

        # 新增：还款意愿-逃避行为平衡调整（核心优化）
        evasion_keywords = {"无法接通", "语音留言", "不说话", "无法接听", "转接留言", "正忙"}
        evasion_flag = 1.0 if any(kw in text for kw in evasion_keywords) else 0.0  # 逃避行为标记
        # 计算平衡特征：意愿强度 - 风险因子
        intent_log_offset = np.float64(0.01)
        intent_strength = (R - D) * math.log(text_length + intent_log_offset) if text_length > 0 else 0.0  # 意愿强度：还款-延迟差×文本长度对数
        evasion_risk_coef = np.float64(0.01)
        risk_factor = O + evasion_flag * evasion_risk_coef  # 风险因子：逾期关键词+逃避标记
        ribs = intent_strength - risk_factor  # 还款意愿-逃避平衡特征
        # 分段复合函数调整：正向放大、负向衰减
        balance_weight = np.float64(2.0)
        log_ribs_offset = np.float64(0.01)
        exp_ribs_gamma = np.float64(2.1779323143313047)
        exp_ribs_gamma = np.float64(1.0)
        positive_balance_multiplier = np.float64(0.1)
        negative_balance_multiplier = np.float64(2.0)
        if ribs > 0:
            balance_adjustment = math.log(ribs + log_ribs_offset) * balance_weight * positive_balance_multiplier  # 正向信号对数放大
        else:
            balance_adjustment = math.exp(ribs / exp_ribs_gamma) * balance_weight * negative_balance_multiplier  # 负向信号指数衰减
        score_before_normalized += balance_adjustment  # 整合平衡调整项

        # 新增：还款意愿-逾期风险-逃避行为三维交互特征（核心优化）
        text_len_floor = np.float64(0.1)
        text_len_log = math.log(max(text_length, text_len_floor))  # 避免log(0)，确保文本长度至少为1
        repayment_overdue_balance = (R - D) * O * text_len_log - evasion_flag * O  # 意愿差×逾期风险×文本长度 - 逃避×逾期风险
        # 复合函数增强区分度：正向用对数放大，负向用指数衰减
        new_balance_weight = np.float64(1.205769484197677)
        log_balance_offset = np.float64(1.0)
        exp_balance_gamma = np.float64(3.043119846880128)
        if repayment_overdue_balance > 0:
            new_balance_adjustment = new_balance_weight * math.log(repayment_overdue_balance + log_balance_offset)
        else:
            new_balance_adjustment = -new_balance_weight * math.exp(repayment_overdue_balance / exp_balance_gamma)
        score_before_normalized += new_balance_adjustment

        # 新增：还款意愿-逃避行为平衡复合特征（关键优化）
        evasion_keywords_ext = {"无法接通", "语音留言", "不说话", "无法接听", "转接留言", "正忙", "挂电话", "不接"}
        evasion_count = sum(1 for kw in evasion_keywords_ext if kw in text)
        evasion_severity = evasion_count + (np.float64(0.1) if "无法接通" in text else np.float64(0.0))  # 逃避严重度：关键词次数+无法接通额外权重
        text_len_log = math.log(max(text_length, np.float64(10.0)))  # 文本长度对数（参数化下限）
        intent_strength = (R - D) * text_len_log  # 还款意愿强度
        # 复合调整函数（参数化系数）
        if intent_strength > 0:
            composite_adjustment = math.log(intent_strength + np.float64(1.0)) * np.float64(2.0) * (1 / (1 + math.exp(-evasion_severity)))
        else:
            composite_adjustment = -math.exp(intent_strength / np.float64(2.4627822490705236)) * np.float64(1.0) * evasion_severity
        score_before_normalized += composite_adjustment * np.float64(0.1)  # 整合调整项

        # 新增：意愿-风险非对称增强特征（关键调整）
        evasion_keywords = {"无法接通", "语音留言", "不说话", "无法接听", "转接留言", "正忙"}
        evasion_flag = 1.0 if any(kw in text for kw in evasion_keywords) else 0.0  # 逃避行为标记
        # 计算平衡特征：意愿强度 - 风险因子
        risk_factor_weighted = O + evasion_flag * np.float64(9.304006758191473)  # 逾期词+加权逃避标记（增强负类风险锚定）
        intention_strength = (R - D) * math.log(text_length + np.float64(8.083122591848525)) if text_length > 0 else np.float64(-3.665962434895765)  # 意愿强度：差值×文本长度对数（避免零值）
        intention_risk_balance = intention_strength - risk_factor_weighted  # 意愿-风险平衡得分
        # 非对称复合函数：正向对数放大，负向指数衰减
        if intention_risk_balance > 0:
            asym_adjustment = math.log(intention_risk_balance + np.float64(0.8056353561301233)) * np.float64(1.7442065744735582) * np.float64(4.473539092600891)  # 正向放大
        else:
            asym_adjustment = -math.exp(intention_risk_balance / np.float64(1.947043582971755)) * np.float64(1.7442065744735582) * np.float64(1.083291061412145)  # 负向抑制
        score_before_normalized += asym_adjustment

        # 新增：意愿-逃避不对称增强特征（核心优化）
        # 1. 意愿强度：还款词减延迟词的差乘以文本长度对数（平滑处理避免零值，增强区分度）
        willingness_intensity = math.log((R - D) / (text_length + np.float64(10.0)) + np.float64(0.1)) if text_length > 0 else 0.0  # 归一化处理，文本越长意愿信号越平滑
        # 2. 逃避严重度：加权逃避关键词计数（高风险关键词权重更高）
        evasion_severity_dict = {"无法接通": np.float64(0.1), "语音留言": np.float64(0.1), "不说话": np.float64(0.1), "无法接听": np.float64(7.870500535701254), "转接留言": np.float64(0.1), "正忙": np.float64(10.0)}
        evasion_severity = sum(evasion_severity_dict[kw] for kw in evasion_severity_dict if kw in text)
        # 3. 不对称复合变换：正向信号对数放大，负向信号指数衰减
        asym_enhance_weight = np.float64(0.1)
        log_enhance_offset = np.float64(2.378988769152897)
        exp_enhance_gamma = np.float64(10.0)
        if willingness_intensity > evasion_severity:
            asym_enhance = math.log((willingness_intensity - evasion_severity) + log_enhance_offset) * asym_enhance_weight
        else:
            asym_enhance = -math.exp(evasion_severity / exp_enhance_gamma) * willingness_intensity * asym_enhance_weight
        # 4. 整合到总分数
        score_before_normalized += asym_enhance

        # 新增：意愿- engagement 与风险-逃避不对称增强特征（关键优化）
        evasion_keywords = {"无法接通", "语音留言", "不说话", "无法接听", "转接留言", "正忙"}
        evasion_flag = 1.0 if any(kw in text for kw in evasion_keywords) else 0.0
        # 1. 意愿-engagement得分：还款意愿差×文本长度对数（强化高 engagement 的正类信号）
        log_engagement_offset = np.float64(0.1)
        willingness_engagement_default = np.float64(1.0)
        willingness_engagement = (R - D) * math.log(text_length + log_engagement_offset) if text_length > 0 else willingness_engagement_default
        # 2. 风险-逃避得分：逾期风险+逃避标记×文本长度倒数（强化低 engagement 的负类信号）
        risk_escapism_log_offset = np.float64(10.0)
        risk_escapism_else_multiplier = np.float64(0.1)
        risk_escapism = (O + evasion_flag) * (1.0 / (text_length + risk_escapism_log_offset)) if text_length > 0 else (O + evasion_flag) * risk_escapism_else_multiplier
        # 3. 不对称复合变换：正向对数放大、负向指数衰减
        engagement_asym_weight = np.float64(0.01)
        log_willingness_engagement_offset = np.float64(2.0)
        positive_engagement_scaler = np.float64(10.0)
        risk_escapism_exp_coef = np.float64(5.0)
        negative_engagement_scaler = np.float64(5.0)
        if willingness_engagement > 0:
            engagement_adjustment = math.log(willingness_engagement + log_willingness_engagement_offset) * positive_engagement_scaler
        else:
            engagement_adjustment = -math.exp(risk_escapism * risk_escapism_exp_coef) * negative_engagement_scaler
        score_before_normalized += engagement_adjustment * engagement_asym_weight

        # 新增：基于参与度与承诺度的非线性复合调整（提升正负类区分度）
        evasion_keywords_final = {"无法接通", "语音留言", "不说话", "无法接听", "转接留言", "正忙", "挂电话", "不接", "听不到"}
        evasion_count_final = sum(1 for kw in evasion_keywords_final if kw in text)
        # 参与度得分：文本长度+句子数（正向）-逃避计数（负向），归一化处理
        engagement_text_weight = np.float64(0.001)
        engagement_sentence_weight = np.float64(2.0)
        engagement_evasion_weight = np.float64(0.001)
        engagement_score = (text_length * engagement_text_weight + sentence_count * engagement_sentence_weight) - (evasion_count_final * engagement_evasion_weight)
        # 承诺度得分：还款意愿差×逾期感知（强化有逾期意识的还款意愿）
        commitment_coef = np.float64(2.0)
        commitment_score = (R - D) * O * commitment_coef
        # 复合非线性变换：Logistic函数放大正负差异，参数化控制强度
        composite_input = engagement_score + commitment_score
        logistic_strength = np.float64(8.669050429249145)
        adjustment_logistic_coef = np.float64(5.0)
        adjustment_logistic = adjustment_logistic_coef * (1 / (1 + math.exp(-logistic_strength * composite_input)) - 0.5)
        score_before_normalized += adjustment_logistic

        # 新增：意愿-逃避强度非对称增强特征（核心优化）
        # 1. 计算加权逃避严重度（高风险逃避关键词加权求和，区分逃避强度）
        evasion_severity_weighted = 0.0
        if "无法接通" in text: evasion_severity_weighted += np.float64(1.7593031106257913)  # 最高风险逃避
        if "语音留言" in text: evasion_severity_weighted += np.float64(4.875157537363745)  # 中高风险逃避
        if any(kw in text for kw in ["哦", "嗯", "啊"]): evasion_severity_weighted += np.float64(4.949875694050333)  # 低风险逃避
        # 2. 计算意愿-逃避平衡得分：意愿强度减去逃避严重度，用对数平滑文本长度
        text_len_smooth = text_length + np.float64(0.39988022404442547)  # 避免log(0)
        intent_escapism_balance = (R - D) * math.log(text_len_smooth) - evasion_severity_weighted if text_len_smooth > 0 else np.float64(-1.6990592524884605)  # 负向默认值抑制无文本样本
        # 3. 非对称Logistic变换：正向放大正类信号，负向抑制负类干扰
        logistic_k = np.float64(0.587371474287472)  # 变换强度
        logistic_x0 = np.float64(-0.6072006550834752)  # 变换中心
        logistic_output = 1 / (1 + math.exp(-logistic_k * (intent_escapism_balance - logistic_x0)))  # Sigmoid变换
        # 4. 整合到总分：正向信号增强，负向信号削弱
        asym_strength_weight = np.float64(1.3927891459426291)  # 调整项权重
        score_before_normalized += asym_strength_weight * (logistic_output - 0.5) * np.float64(3.1191598457445635)  # 对称化处理增强区分度

        # 新增：逃避行为-意愿-逾期三维平衡非对称增强特征（关键优化）
        evasion_keywords_core = {"无法接通", "语音留言", "无法接听", "转接留言"}  # 核心逃避关键词
        evasion_core_flag = 1.0 if any(kw in text for kw in evasion_keywords_core) else 0.0  # 核心逃避标记
        # 1. 计算意愿-逾期差异：增强对「高意愿vs高逾期」的区分
        willingness_overdue_diff = abs((R - D) - np.float64(1.2852475597765383) * O)  # 意愿差与逾期感知的绝对差异
        log_diff = math.log(np.float64(4.169401208279867) + willingness_overdue_diff) if willingness_overdue_diff > 0 else np.float64(0.28469133188416107)  # 对数平滑增强区分度
        # 2. 构建逃避影响权重：sigmoid函数结合文本长度，抑制长文本逃避干扰
        evasion_impact_weight = 1.0 / (1 + math.exp(-(evasion_core_flag * np.float64(6.523392054656825) - text_length * np.float64(0.16544229372359492))))  # 逃避标记增强、文本长度削弱
        # 3. 复合调整项：对数差异×逃避权重，强化负类信号抑制
        asym_balance_coef = np.float64(0.8216752873310765)  # 调整项权重
        evasion_willingness_balance = log_diff * evasion_impact_weight * asym_balance_coef
        score_before_normalized -= evasion_willingness_balance  # 负向调整增强负类区分（核心逃避+高差异=更强负向信号）

        # 新增：非对称还款-逃避平衡特征（提升AUC核心调整）
        # 1. 计算核心逃避严重度（高风险关键词加权）
        core_evasion_keywords = {"无法接通": np.float64(2.208891336266139), "语音留言": np.float64(0.1), "无法接听": np.float64(5.0), "转接留言": np.float64(0.1), "哦": np.float64(0.01), "嗯": np.float64(0.01), "啊": np.float64(0.01)}
        evasion_severity = sum(core_evasion_keywords[kw] for kw in core_evasion_keywords if kw in text)
        # 2. 计算基础平衡得分：意愿差×文本长度对数（平滑）- 逃避严重度
        text_len_smooth_offset = np.float64(7.75533217782561)  # 平滑文本长度
        text_len_smooth = text_length + text_len_smooth_offset
        base_balance = (R - D) * math.log(text_len_smooth) - evasion_severity
        # 3. 非对称复合变换：正向对数放大正类，负向指数衰减负类
        asym_weight = np.float64(1.0)
        log_offset = np.float64(1.0)
        exp_gamma = np.float64(1.5444847757995448)
        positive_multiplier = np.float64(2.0)
        negative_multiplier = np.float64(1.1720999831332435)
        if base_balance > 0:
            asym_adjustment = asym_weight * math.log(base_balance + log_offset) * positive_multiplier
        else:
            asym_adjustment = -asym_weight * math.exp(base_balance / exp_gamma) * negative_multiplier
        score_before_normalized += asym_adjustment

        # 新增：逃避意愿不对称增强指数（AUC提升关键特征）
        # 1. 计算加权逃避严重度（高风险词高权重）
        evasion_severity_ew = 0.0
        high_evasion_words = {"无法接通": np.float64(7.834678064820691), "语音留言": np.float64(5.499845732300528), "无法接听": np.float64(16.591876432123396), "转接留言": np.float64(7.199391201202426)}
        low_evasion_words = {"哦": np.float64(2.8165357517769336), "嗯": np.float64(5.431533870750902), "啊": np.float64(1.4178330074978789), "过": np.float64(8.023947837732857)}
        evasion_severity_ew += sum(high_evasion_words[kw] for kw in high_evasion_words if kw in text)
        evasion_severity_ew += sum(low_evasion_words[kw] for kw in low_evasion_words if kw in text)
        # 2. 计算意愿强度（平滑处理）
        log_willingness_offset_ew = np.float64(0.8380513724297312)
        willingness_default_ew = np.float64(-0.13113063399482705)
        willingness_strength_ew = (R - D) * math.log(text_length + log_willingness_offset_ew) if text_length > 0 else willingness_default_ew
        # 3. 计算文本参与度（文本长度+句子数，归一化）
        engagement_normalizer_ew = np.float64(774.5223216036909)
        engagement_ew = (text_length + sentence_count) / engagement_normalizer_ew
        # 4. 构建交互特征：意愿强度 - 逃避严重度×参与度倒数（强化逃避在低参与度中的影响）
        div_zero_offset_ew = np.float64(0.2067285247188307)
        interaction_ew = willingness_strength_ew - (evasion_severity_ew * (1.0 / (engagement_ew + div_zero_offset_ew)))
        # 5. 不对称复合变换：正向对数放大、负向指数衰减
        ew_weight = np.float64(0.03755536444677597)
        log_offset_ew = np.float64(0.8173068141702858)
        exp_gamma_ew = np.float64(7.097887704091409)
        if interaction_ew > 0:
            ew_adjustment = ew_weight * math.log(interaction_ew + log_offset_ew)
        else:
            ew_adjustment = -ew_weight * math.exp(interaction_ew / exp_gamma_ew)
        score_before_normalized += ew_adjustment

        # 新增：意愿-逃避-参与度三维复合logistic调整（核心AUC提升特征）
        # 1. 意愿参与度：还款-延迟差×文本长度对数（强化详细文本的正类信号）
        willingness_engagement = (R - D) * math.log(text_length + np.float64(0.1)) if text_length > 0 else np.float64(0.0)  # 避免log(0)，短文本惩罚
        # 2. 逃避影响度：高风险逃避词加权×句子数负指数（弱化高参与度文本的逃避干扰）
        high_evasion = {"无法接通": np.float64(10.0), "语音留言": np.float64(10.0), "无法接听": np.float64(10.0), "转接留言": np.float64(10.0)}
        evasion_impact = sum(high_evasion[kw] for kw in high_evasion if kw in text) * math.exp(-sentence_count / np.float64(20.0))
        # 3. 三维交互指数：意愿参与度-逃避影响度（强化正类、抑制负类）
        engagement_index = willingness_engagement - evasion_impact
        # 4. Logistic非对称变换：放大正负差异（参数化强度与中心）
        logistic_adjustment = np.float64(0.1) * (
            np.float64(10.0) / (
                np.float64(0.1) + math.exp(-np.float64(20.0) * (engagement_index - np.float64(5.0)))
            ) - np.float64(0.1)
        )  # 对称化调整
        score_before_normalized += logistic_adjustment  # 整合到总分

        # 新增：还款意愿-逃避行为复合增强特征（AUC提升关键）
        # 1. 计算意愿-逃避平衡得分：（还款词减延迟词差）×文本长度对数（平滑）- 加权逃避严重度
        evasion_keywords_ext = {"无法接通", "语音留言", "不说话", "无法接听", "转接留言", "正忙", "挂电话", "不接"}
        evasion_severity_final = sum(np.float64(10.0) if kw in {"无法接通", "无法接听"} else np.float64(5.0) for kw in text.split() if kw in evasion_keywords_ext)  # 高风险逃避词高权重
        text_log_smooth = math.log(text_length + np.float64(0.01)) if text_length > 0 else np.float64(0.01)  # 避免log(0)
        willingness_escapism_balance = (R - D) * text_log_smooth - evasion_severity_final
        # 2. 非对称复合变换：正向用对数放大，负向用指数衰减
        composite_weight = np.float64(0.01)
        log_offset_final = np.float64(2.0)
        exp_gamma_final = np.float64(3.40174375441192)
        if willingness_escapism_balance > 0:
            composite_adjustment_final = composite_weight * math.log(willingness_escapism_balance + log_offset_final) * np.float64(0.1)
        else:
            composite_adjustment_final = -composite_weight * math.exp(willingness_escapism_balance / exp_gamma_final) * np.float64(5.0)
        score_before_normalized += composite_adjustment_final

        # 新增：意愿-逾期-逃避三维非对称增强特征（AUC核心优化）
        # 1. 逃避严重度（ES）：语音留言强标记+核心逃避关键词计数+文本长度归一化
        is_voice_message = 1.0 if any(kw in text for kw in ["语音留言", "无法接听", "通话已转至语音", "尝试联系的用户无法接听"]) else 0.0
        core_evasion_count = sum(1 for kw in ["无法接通", "语音留言", "哦", "嗯", "啊"] if kw in text)
        es_base = is_voice_message * np.float64(79.66801378877248) + core_evasion_count * np.float64(6.435767700281267)
        es_normalized = es_base / (text_length + np.float64(10.0)) if text_length > 0 else es_base
        # 2. 意愿-逾期-参与度交互（WOE）：还款意愿差×逾期感知×文本参与度
        text_engagement = text_length + sentence_count
        woe = (R - D) * O * text_engagement
        # 3. 非对称复合调整：正向对数放大+负向指数衰减
        asym_weight_total = np.float64(0.01)
        log_woe_offset = np.float64(0.001)
        exp_es_gamma = np.float64(0.001)
        if woe > 0:
            woe_adjustment = asym_weight_total * math.log(woe + log_woe_offset)
        else:
            woe_adjustment = np.float64(1.0)
        if es_normalized > 0:
            es_adjustment = -asym_weight_total * math.exp(es_normalized * exp_es_gamma)
        else:
            es_adjustment = np.float64(1.0)
        score_before_normalized += woe_adjustment + es_adjustment

        # 新增：还款意愿-逃避行为不对称增强特征（AUC提升关键）
        # 1. 提取逃避关键词并计算加权严重度（高风险逃避词权重更高）
        evasion_keywords = {
            "无法接通": np.float64(0.9711458824349786),
            "语音留言": np.float64(7.191604771139313),
            "机主忙": np.float64(7.820267054445544),
            "转接留言": np.float64(4.637288952907755),
            "不说话": np.float64(7.145474607726963),
            "挂断": np.float64(9.025432164485034)
        }
        evasion_severity = sum(evasion_keywords[kw] for kw in evasion_keywords if kw in text)
        # 2. 计算意愿-逃避平衡得分：(还款词差×文本长度对数) - 逃避严重度×平衡系数（平衡信号强度）
        text_log = math.log(text_length + np.float64(5.890540735626166)) if text_length > 0 else np.float64(0.6993369398266747)  # 避免log(0)，平滑处理
        willingness_balance = (R - D) * text_log - evasion_severity * np.float64(1.272317760659874)
        # 3. 不对称Sigmoid变换：正向放大正类、负向抑制负类（参数化变换强度与对称化偏移）
        asym_adjustment = 1 / (1 + math.exp(-np.float64(19.53479690318953) * willingness_balance)) - np.float64(0.5158558171872614)  # 对称化处理增强区分度
        # 4. 整合到总分（参数化调整项权重避免过拟合）
        score_before_normalized += asym_adjustment * np.float64(0.5628077284868251)

        # 新增：承诺-逃避不对称Logistic增强特征（核心AUC提升）
        # 1. 承诺强度：还款词减延迟词差×文本长度对数（平滑处理避免零值）
        log_text_commitment_offset = np.float64(-10.0)
        commitment_default = np.float64(-10.0)
        commitment_strength = (R - D) * math.log(text_length + log_text_commitment_offset) if text_length > 0 else commitment_default  # 文本越长承诺信号越强
        # 2. 逃避严重度：高风险逃避词加权和（无法接通/无法接听高权重，哦嗯啊低权重）
        evasion_severity = 0.0
        high_evasion_weight = np.float64(41.720447816315385)
        if "无法接通" in text or "无法接听" in text: evasion_severity += high_evasion_weight  # 高风险逃避
        low_evasion_weight = np.float64(0.01)
        if any(kw in text for kw in ["哦", "嗯", "啊"]): evasion_severity += low_evasion_weight  # 低风险逃避
        # 3. 差异特征：承诺强度 - 逃避严重度（正向=高承诺低逃避，负向=低承诺高逃避）
        commitment_evasion_diff = commitment_strength - evasion_severity
        # 4. 强不对称Logistic变换：放大正负差异（参数化强度与中心）
        logistic_strength = np.float64(0.01)
        logistic_center = np.float64(-10.0)
        logistic_adjust_weight = np.float64(0.01)
        logistic_adjust = logistic_adjust_weight * (
            1 / (1 + math.exp(-logistic_strength * (commitment_evasion_diff - logistic_center))) - 0.5
        )
        score_before_normalized += logistic_adjust

        # 新增：「还款意愿-逾期感知-逃避行为」三维不对称复合特征（核心AUC优化）
        # 1. 计算意愿-逾期-参与度交互得分：还款意愿差×逾期感知×文本长度对数（平滑处理，强化高参与度正类信号）
        text_len_log_smooth = math.log(text_length + np.float64(7.564395329336954)) if text_length > 0 else np.float64(0.21576267593583856)  # 避免log(0)，文本长度≥1
        willingness_overdue_engagement = (R - D) * O * text_len_log_smooth
        # 2. 计算逃避严重度：高风险逃避词加权和（区分逃避强度）
        high_evasion_severity = sum(
            np.float64(0.961108796881893) if kw in {"无法接通", "无法接听", "语音留言"} else  # 高风险逃避
            np.float64(40.45748389329175) if kw in {"哦", "嗯", "啊", "过"} else  # 中风险逃避
            np.float64(7.137682039132624)  # 低风险逃避
            for kw in text.split()
            if kw in evasion_keywords_final
        )
        # 3. 构建平衡特征：意愿-逾期-参与度交互 - 逃避严重度（正向=高意愿高感知低逃避，负向=低意愿低感知高逃避）
        tri_dim_balance = willingness_overdue_engagement - high_evasion_severity
        # 4. 不对称复合变换：正向用对数放大正类信号，负向用指数衰减抑制负类干扰
        if tri_dim_balance > 0:
            tri_dim_adjustment = np.float64(1.3723528430242273) * math.log(tri_dim_balance + np.float64(3.437271408077559)) * np.float64(5.972229448915981)  # 正向放大
        else:
            tri_dim_adjustment = -np.float64(1.3723528430242273) * math.exp(tri_dim_balance / np.float64(1.362899097856382)) * np.float64(1.6357687264103744)  # 负向抑制
        score_before_normalized += tri_dim_adjustment

        # 新增：低阶逃避词-模糊承诺词不对称调整（核心AUC提升）
        # 1. 计算低阶逃避词频率（"啊"/"嗯"/"哦"）归一化得分
        low_evasion_words = {"啊", "嗯", "哦"}
        low_evasion_count = sum(1 for word in text if word in low_evasion_words)
        text_len_norm = max(text_length, np.float64(2.664116906514115))  # 避免除以零，参数化下限
        ess = low_evasion_count / text_len_norm  # 低阶逃避严重度
        # 2. 计算模糊承诺词频率（"稍微等一下"/"过一下"/"等一下"）归一化得分
        vague_commitment_words = {"稍微等一下", "过一下", "等一下"}
        vague_commitment_count = sum(1 for word in text if word in vague_commitment_words)
        vcs = vague_commitment_count / text_len_norm  # 模糊承诺得分
        # 3. 不对称复合调整：强 penalize 高ESS，弱 penalize 高VCS，温和 reward 低VCS
        if ess > np.float64(0.0628957184512023):  # 高逃避频率阈值（参数化）
            ess_penalty = -np.float64(4.857282958357695) * math.exp(ess * np.float64(16.056784077006366))  # 指数衰减放大负向惩罚（参数化系数）
        else:
            ess_penalty = np.float64(-0.077480618925429)  # 低逃避无惩罚（参数化默认值）
        if vcs > np.float64(0.07847239788930546):  # 高模糊承诺阈值（参数化）
            vcs_penalty = -np.float64(0.2548444497876525) * vcs  # 线性削弱弱正向干扰（参数化系数）
        else:
            vcs_reward = np.float64(0.6768262808994174) * math.log(vcs + np.float64(6.504432661274378)) if vcs > 0 else np.float64(0.08487281009787256)  # 对数增长奖励低模糊承诺（参数化系数和偏移）
            vcs_penalty = vcs_reward
        score_before_normalized += ess_penalty + vcs_penalty

        prob = 1 / (1 + math.exp(-score_before_normalized))
        prob = max(0.0, min(1.0, prob))

        return score_before_normalized, prob
