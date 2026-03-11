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
自动生成的代码展现了对复杂字符串的分隔、关键词统计以及非线性复合算子（如 `sqrt` 与 `log` 的交互）的运用。

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
