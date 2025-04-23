# lib/selfcheck.py

import torch
import re
from typing import List, Tuple, Dict, Optional

class SelfCheckVerifier:
    """
    实现基于SelfCheck的验证机制，用于评估和改进推理步骤
    """
    def __init__(self, tokenizer, modules, args):
        self.tokenizer = tokenizer
        self.modules = modules
        self.args = args
    
    def extract_goal(self, question: str, current_step_idx: int, previous_steps: List[str]) -> str:
        """
        提取当前步骤的目标
        
        Args:
            question: 原始问题
            current_step_idx: 当前是第几个推理步骤
            previous_steps: 之前的所有推理步骤
            
        Returns:
            goal: 当前步骤的目标描述
        """
        # 构建提示来提取目标
        prompt = f"""
问题: {question}

已有的推理步骤:
{' '.join(previous_steps)}

考虑到上述信息，第{current_step_idx+1}步推理的目标是什么?
请提供一个简明的描述，说明这一步需要解决的具体问题。
        """
        
        # 这里使用模型生成目标描述
        # 在实际实现中，可以使用更高效的方法
        goal = self._generate_text(prompt)
        return goal
    
    def filter_relevant_context(self, 
                               question: str, 
                               goal: str,
                               previous_steps: List[str], 
                               k: int = 3) -> List[str]:
        """
        筛选与当前目标最相关的上下文信息
        
        Args:
            question: 原始问题
            goal: 当前步骤的目标
            previous_steps: 之前的所有推理步骤
            k: 最多选择多少个相关步骤
            
        Returns:
            relevant_steps: 最相关的k个步骤
        """
        if not previous_steps:
            return []
            
        if len(previous_steps) <= k:
            return previous_steps
            
        # 计算每个步骤与目标的相关性
        relevance_scores = []
        for i, step in enumerate(previous_steps):
            # 构建提示来评估相关性
            prompt = f"""
问题: {question}

推理步骤: {step}

目标: {goal}

请评估上述推理步骤与目标的相关性，给出1-10的评分。
1表示完全不相关，10表示极其相关。
只需要输出数字评分。
            """
            # 生成评分
            score_text = self._generate_text(prompt)
            try:
                # 尝试提取数字
                score = float(re.search(r'(\d+(?:\.\d+)?)', score_text).group(1))
                relevance_scores.append((i, score))
            except:
                # 如果无法提取，给一个默认分数
                relevance_scores.append((i, 5.0))
        
        # 按相关性排序并选择前k个
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in relevance_scores[:k]]
        selected_indices.sort()  # 保持原始顺序
        
        return [previous_steps[i] for i in selected_indices]
    
    def regenerate_step(self, 
                       question: str, 
                       goal: str, 
                       relevant_context: List[str]) -> str:
        """
        基于筛选后的上下文重新生成推理步骤
        
        Args:
            question: 原始问题
            goal: 当前步骤的目标
            relevant_context: 筛选后的相关上下文
            
        Returns:
            regenerated_step: 重新生成的推理步骤
        """
        context_str = "\n".join(relevant_context)
        
        prompt = f"""
问题: {question}

相关上下文:
{context_str}

目标: {goal}

基于上述信息，请生成下一个推理步骤:
        """
        
        regenerated_step = self._generate_text(prompt)
        return regenerated_step
    
    def compare_steps(self, 
                     original_step: str, 
                     regenerated_step: str, 
                     question: str,
                     goal: str) -> Tuple[float, str]:
        """
        比较原始步骤和重新生成的步骤，评估正确性
        
        Args:
            original_step: 原始生成的推理步骤
            regenerated_step: 重新生成的推理步骤
            question: 原始问题
            goal: 当前步骤的目标
            
        Returns:
            confidence: 置信度分数 (0-1)
            final_step: 最终确定的步骤
        """
        # 构建比较提示
        prompt = f"""
问题: {question}

目标: {goal}

步骤A: {original_step}

步骤B: {regenerated_step}

比较上述两个推理步骤，哪一个更准确、更有助于解决问题？
请给出你的分析，并做出以下判断:
1. 步骤A的正确率 (0-100%)
2. 步骤B的正确率 (0-100%)
3. 最终推荐使用哪个步骤 (A、B或C-混合两者的优点)

如果推荐C，请提供一个结合两个步骤优点的新版本。
        """
        
        comparison = self._generate_text(prompt)
        
        # 提取置信度和最终步骤
        try:
            # 提取A的置信度
            confidence_a = float(re.search(r'A.*?(\d+)%', comparison).group(1)) / 100.0
        except:
            confidence_a = 0.7  # 默认值
            
        try:
            # 提取选择哪个步骤
            choice = re.search(r'推荐使用.*?([ABC])', comparison).group(1)
        except:
            choice = 'A'  # 默认使用原始步骤
        
        # 根据选择决定最终步骤
        if choice == 'A':
            final_step = original_step
            confidence = confidence_a
        elif choice == 'B':
            final_step = regenerated_step
            confidence = 1.0 - confidence_a  # B的置信度
        else:  # C - 混合
            # 尝试提取混合步骤
            try:
                mixed_step = re.search(r'C.*?((?:.|\n)+?)(?:$|推荐|最终)', comparison).group(1).strip()
                final_step = mixed_step
                confidence = 0.8  # 假设混合步骤的置信度较高
            except:
                final_step = original_step
                confidence = confidence_a
        
        return confidence, final_step
    
    def verify_step(self, 
                   question: str, 
                   current_step: str, 
                   previous_steps: List[str],
                   current_step_idx: int) -> Tuple[float, str]:
        """
        验证当前推理步骤并提供改进
        
        Args:
            question: 原始问题
            current_step: 当前推理步骤
            previous_steps: 之前的推理步骤列表
            current_step_idx: 当前是第几个推理步骤
            
        Returns:
            confidence: 置信度分数 (0-1)
            verified_step: 经过验证/修正的步骤
        """
        # 1. 提取当前步骤的目标
        goal = self.extract_goal(question, current_step_idx, previous_steps)
        
        # 2. 筛选相关上下文
        relevant_context = self.filter_relevant_context(
            question, 
            goal, 
            previous_steps, 
            k=min(3, len(previous_steps))
        )
        
        # 3. 独立再生成步骤
        regenerated_step = self.regenerate_step(question, goal, relevant_context)
        
        # 4. 比较并选择最终步骤
        confidence, verified_step = self.compare_steps(
            current_step,
            regenerated_step,
            question,
            goal
        )
        
        return confidence, verified_step
    
    def _generate_text(self, prompt: str) -> str:
        """
        使用模型生成文本的辅助方法
        实际实现时需要根据具体模型调整
        """
        # 这里简化了生成过程，实际实现需要调用模型API
        # 可以使用diffusion生成，也可以调用外部LLM API
        
        # 简单模拟生成
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens]).cuda()
        src_mask = torch.ones_like(input_ids, dtype=bool).cuda()
        
        # 使用diffusion模型生成
        with torch.no_grad():
            output_ids = generate_samples(
                input_ids, 
                src_mask, 
                self.modules, 
                self.args
            )
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(output_ids[0].tolist())
        
        # 提取回答部分
        return generated_text