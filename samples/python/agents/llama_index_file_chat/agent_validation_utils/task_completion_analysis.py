import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskCompletionAnalyzer:
    task_data = []  # 静态变量，所有实例共享

    def __init__(self):
        pass  # 不再需要初始化 task_data
    
    def add_task_result(self, task: str, task_id: str, is_complete: bool, reason: str = ""):
        """
        添加任务执行结果
        :param task: 任务类型
        :param task_id: 任务ID
        :param is_complete: 是否完成
        :param reason: 任务完成或未完成的原因
        """
        TaskCompletionAnalyzer.task_data.append({
            'task': task,
            'task_id': task_id,
            'is_complete': is_complete,
            'reason': reason
        })
    
    def generate_pie_chart(self, output_path='agent_validation_utils/report/task_completion_pie.png'):
        """
        Generate a pie chart of task completion status
        """
        try:
            df = pd.DataFrame(TaskCompletionAnalyzer.task_data)
            completion_counts = df['is_complete'].value_counts()
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            logger.info(f"Creating directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建标签映射
            label_map = {True: 'Completed', False: 'Incomplete'}
            labels = [label_map[idx] for idx in completion_counts.index]
            
            # 确保颜色顺序与标签顺序一致
            colors = ['#2ecc71' if label == 'Completed' else '#e74c3c' for label in labels]
            
            plt.figure(figsize=(10, 8))
            plt.pie(completion_counts, 
                    labels=labels,
                    autopct='%1.1f%%',
                    colors=colors,  # 使用新的颜色列表
                    startangle=90)
            
            plt.title('Mobile-RPA-Agent Task Completion Statistics')
            plt.axis('equal')
            
            logger.info(f"Saving pie chart to: {output_path}")
            plt.savefig(output_path)
            plt.close()
            logger.info("Pie chart saved successfully")
            
        except Exception as e:
            logger.error(f"Error generating pie chart: {str(e)}")
            raise
    
    def analyze_task_types(self):
        """
        分析不同任务类型的完成情况
        """
        df = pd.DataFrame(TaskCompletionAnalyzer.task_data)
        task_type_stats = df.groupby(['task', 'is_complete']).size().unstack(fill_value=0)
        
        # 确保所有可能的列都存在
        if True not in task_type_stats.columns:
            task_type_stats[True] = 0
        if False not in task_type_stats.columns:
            task_type_stats[False] = 0
            
        # 重命名列
        task_type_stats.columns = ['已完成' if col else '未完成' for col in task_type_stats.columns]
        
        # 计算完成率
        task_type_stats['完成率'] = task_type_stats['已完成'] / (task_type_stats['已完成'] + task_type_stats['未完成']) * 100
        return task_type_stats

    def get_task_details(self):
        """
        获取所有任务的详细信息，包括完成情况和原因
        """
        df = pd.DataFrame(TaskCompletionAnalyzer.task_data)
        return df[['task', 'task_id', 'is_complete', 'reason']]

    def save_stats_to_csv(self, output_path='agent_validation_utils/report/task_completion_stats.csv'):
        """
        保存统计结果到CSV文件
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        task_stats = self.analyze_task_types()
        task_stats.to_csv(output_path, encoding='utf-8-sig')
        return task_stats

    def save_task_details_to_csv(self, output_path='agent_validation_utils/report/task_details.csv'):
        """
        保存任务详细信息到CSV文件
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        task_details = self.get_task_details()
        task_details.to_csv(output_path, encoding='utf-8-sig')
        return task_details

# 使用示例
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = TaskCompletionAnalyzer()
    
    # 示例：添加一些任务结果
    analyzer.add_task_result("Task A", "001", True, "Successfully completed")
    analyzer.add_task_result("Task A", "002", True, "Failed due to error")
    analyzer.add_task_result("Task B", "003", True, "Completed on time")
    analyzer.add_task_result("Task B", "004", True, "Completed with no issues")
    
    # 生成饼图
    analyzer.generate_pie_chart()
    
    # 获取并打印统计信息
    stats = analyzer.save_stats_to_csv()
    print("\n各任务类型完成情况：")
    print(stats)
    
    # 获取并打印任务详细信息
    details = analyzer.save_task_details_to_csv()
    print("\n任务详细信息：")
    print(details) 